"""PC-side HTTP collector for P30 telemetry + event frames.

Stdlib only. Thread-safe appends; oldest frames evicted past the cap.
"""

import csv
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import protocol


class Collector:
    def __init__(self, session_dir, frame_cap_bytes=protocol.FRAME_DIR_CAP_BYTES):
        self.session_dir = session_dir
        self.frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._phone_csv = open(
            os.path.join(session_dir, "phone_telemetry.csv"),
            "a",
            newline="",
            buffering=1,
            encoding="utf-8",
        )
        self._phone_writer = csv.writer(self._phone_csv)
        if os.path.getsize(os.path.join(session_dir, "phone_telemetry.csv")) == 0:
            self._phone_writer.writerow(protocol.PHONE_CSV_HEADER)

        self.last_heartbeat_at = 0.0
        self.last_heartbeat_body = {}
        self.last_inference_fps = 0.0
        self.last_inference_at = 0.0
        self.last_row = {}
        self.last_row_at = 0.0
        self.frames_received = 0
        self.rows_received = 0
        self.stop_requested = False
        self.stop_reason = ""
        self._frame_cap = frame_cap_bytes
        self._server = None

    # -- HTTP plumbing -------------------------------------------------
    def handler_class(self):
        collector = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CatStalkerCollector/1.0"

            def log_message(self, *args):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                if length > 20 * 1024 * 1024:
                    return self._json(413, {"error": "body too large"})
                body = self.rfile.read(length) if length else b""

                if parsed.path == protocol.HELLO_PATH:
                    try:
                        hello = json.loads(body.decode("utf-8")) if body else {}
                    except ValueError:
                        return self._json(400, {"error": "bad json"})
                    collector.note_heartbeat(hello)
                    return self._json(200, {
                        "server_time": time.time(),
                        "stop": collector.stop_requested,
                        "reason": collector.stop_reason,
                    })

                if parsed.path == protocol.TELEMETRY_PATH:
                    try:
                        payload = json.loads(body.decode("utf-8"))
                        rows = payload["rows"]
                    except (ValueError, KeyError, TypeError):
                        return self._json(400, {"error": "bad telemetry payload"})
                    ack = collector.append_telemetry(payload)
                    return self._json(200, {"ack": ack})

                if parsed.path == protocol.FRAME_PATH:
                    qs = parse_qs(parsed.query)
                    meta = {
                        "session_id": (qs.get("session_id") or [""])[0],
                        "kind": (qs.get("kind") or ["event"])[0],
                        "ts": (qs.get("ts") or ["0"])[0],
                        "conf": (qs.get("conf") or [""])[0],
                    }
                    if self.headers.get("Content-Type") != "image/jpeg":
                        return self._json(400, {"error": "expected image/jpeg"})
                    name = collector.store_frame(body, meta)
                    return self._json(200, {"stored": name})

                if parsed.path == "/api/v1/stop":
                    qs = parse_qs(parsed.query)
                    reason = (qs.get("reason") or ["dashboard stop"])[0]
                    try:
                        form = body.decode("utf-8")
                        for part in form.split("&"):
                            if part.startswith("reason="):
                                import urllib.parse as _up

                                reason = _up.unquote_plus(part[len("reason="):])
                    except ValueError:
                        pass
                    collector.stop_requested = True
                    collector.stop_reason = reason
                    return self._json(200, {"stop": True, "reason": reason})

                return self._json(404, {"error": "unknown path"})

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == protocol.SESSION_PATH:
                    return self._json(200, {
                        "server_time": time.time(),
                        "stop": collector.stop_requested,
                        "reason": collector.stop_reason,
                    })
                if parsed.path == "/" or parsed.path == "/status":
                    return self._dashboard()
                return self._json(404, {"error": "unknown path"})

            def _dashboard(self):
                now = time.time()
                hb_age = now - collector.last_heartbeat_at if collector.last_heartbeat_at else -1
                html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Cat Stalker session</title></head><body>
<h1>Cat Stalker session</h1>
<ul>
<li>phone rows: {collector.rows_received}</li>
<li>frames: {collector.frames_received}</li>
<li>last phone heartbeat: {"never" if hb_age < 0 else f"{{hb_age:.0f}}s ago"}</li>
<li>last inference FPS: {collector.last_inference_fps:.1f}</li>
<li>stop requested: {collector.stop_requested} ({collector.stop_reason})</li>
</ul>
<form method="post" action="/api/v1/stop">
<input type="hidden" name="reason" value="dashboard stop">
<button type="submit">Stop session</button></form>
<p>Details: STATUS.md and events.jsonl in the session dir.</p>
</body></html>"""
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def serve(self, host, port):
        self._server = ThreadingHTTPServer((host, port), self.handler_class())
        self._server.daemon_threads = True
        self._server.serve_forever()

    def shutdown(self):
        if self._server is not None:
            self._server.shutdown()
        with self._lock:
            try:
                self._phone_csv.close()
            except Exception:
                pass

    # -- storage -------------------------------------------------------
    def note_heartbeat(self, hello):
        with self._lock:
            self.last_heartbeat_at = time.time()
            self.last_heartbeat_body = dict(hello)

    def append_telemetry(self, payload):
        recv_ts = time.time()
        device_id = str(payload.get("device_id", ""))
        session_id = str(payload.get("session_id", ""))
        count = 0
        with self._lock:
            for row in payload.get("rows", []):
                self._phone_writer.writerow([
                    row.get("ts", ""),
                    f"{recv_ts:.3f}",
                    device_id,
                    row.get("state", ""),
                    row.get("confidence", ""),
                    row.get("offset_x", ""),
                    row.get("area_ratio", ""),
                    row.get("inference_ms", ""),
                    row.get("analysis_fps", ""),
                    row.get("inference_fps", ""),
                    row.get("backend", ""),
                    row.get("battery_pct", ""),
                    row.get("disk_free_mb", ""),
                ])
                count += 1
            self.rows_received += count
            self.last_heartbeat_at = recv_ts
            if payload.get("rows"):
                self.last_row = dict(payload["rows"][-1])
                self.last_row_at = recv_ts
            if session_id:
                self.last_heartbeat_body["session_id"] = session_id
            try:
                fps_rows = payload.get("rows", [])
                if fps_rows:
                    self.last_inference_fps = float(
                        fps_rows[-1].get("inference_fps") or 0.0
                    )
                    self.last_inference_at = recv_ts
            except (TypeError, ValueError):
                pass
        return count

    def fresh_inference_fps(self, max_age_s=5.0):
        """Latest phone inference FPS, or 0.0 when stale."""
        with self._lock:
            if time.time() - self.last_inference_at > max_age_s:
                return 0.0
            return self.last_inference_fps

    def store_frame(self, jpeg_bytes, meta):
        recv_ts = time.time()
        safe_conf = "".join(c for c in meta["conf"] if c.isdigit() or c == ".") or "na"
        name = f"{recv_ts:.3f}_{meta['kind']}_conf{safe_conf}.jpg"
        path = os.path.join(self.frames_dir, name)
        with self._lock:
            with open(path, "wb") as f:
                f.write(jpeg_bytes)
            self.frames_received += 1
            self._evict_old_frames_locked()
        return name

    def _evict_old_frames_locked(self):
        total = 0
        entries = []
        for name in os.listdir(self.frames_dir):
            path = os.path.join(self.frames_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            total += size
            entries.append((os.path.getmtime(path), path, size))
        if total <= self._frame_cap:
            return
        entries.sort()
        for _, path, size in entries:
            try:
                os.remove(path)
                total -= size
            except OSError:
                pass
            if total <= self._frame_cap:
                break
