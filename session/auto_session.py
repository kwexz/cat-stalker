"""Autonomous data-collection session orchestrator.

Phone streams telemetry/frames; PC persists them; Dreame link (when
DREAME_TOKEN is set) keeps Wi-Fi awake, optionally sweeps slowly for
diverse angles, and uses the robot speaker as the human-attention
channel. Ctrl+C always ends with a safe stop (0,0 + stop_clean).

Motion policy:
- observe (default): no motion commands at all, keepalive move(0,0) only.
- patrol: slow in-place rotation sweep (alternating direction) for
  diverse capture angles. Conservative speed, stops on any watchdog
  trip. No translational driving in this stage.

Stuck detection honesty note: without robot odometry/pose we cannot
observe blocked wheels. The watchdog treats persistent link errors
(failures > 0 for >10 s) as link-degraded: patrol halts to keepalive,
attention sounds. True motion-stuck detection needs Dreame pose/map
(a later stage).
"""

import argparse
import csv
import json
import os
import shutil
import socket
import sys
import threading
import time
from datetime import datetime

from . import protocol
from .collector import Collector
from .discovery import DiscoveryResponder
from .robot_link import (
    MAX_FAILURES,
    SIG_ATTENTION,
    SIG_DONE,
    SIG_ERROR,
    RobotLink,
)


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("observe", "patrol", "follow"), default="observe"
    )
    parser.add_argument("--duration-min", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--patrol-rotation", type=int, default=8)
    parser.add_argument("--patrol-interval", type=float, default=20.0)
    parser.add_argument(
        "--analyze-every-min",
        type=float,
        default=0.0,
        help="Run session/analyze.py on the live session dir every N minutes (0 = off).",
    )
    return parser.parse_args(argv)


class Session:
    def __init__(self, args):
        self.args = args
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(args.session_root, stamp)
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)

        self.events_path = os.path.join(self.dir, "events.jsonl")
        self._events = open(self.events_path, "a", buffering=1, encoding="utf-8")
        self._robot_csv = open(
            os.path.join(self.dir, "robot_telemetry.csv"),
            "a",
            newline="",
            buffering=1,
            encoding="utf-8",
        )
        self._robot_writer = csv.writer(self._robot_csv)
        self._robot_writer.writerow(protocol.ROBOT_CSV_HEADER)
        self._status_csv = open(
            os.path.join(self.dir, "robot_status.csv"),
            "a",
            newline="",
            buffering=1,
            encoding="utf-8",
        )
        self._status_writer = csv.writer(self._status_csv)
        self._status_writer.writerow(
            ["recv_ts", "battery_level", "device_fault", "device_status"]
        )
        self._fault_held = False

        with open(os.path.join(self.dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "started_at": stamp,
                    "mode": args.mode,
                    "duration_min": args.duration_min,
                    "patrol_rotation": args.patrol_rotation,
                    "patrol_interval": args.patrol_interval,
                },
                f,
                indent=2,
            )

        self.collector = Collector(self.dir)
        self.robot = None
        self.patrol_scale = 1.0
        self._phone_held = False
        self._stop = threading.Event()
        self._stop_reason = ""
        self._attention_sounded = False
        self._link_degraded_since = 0.0

    def event(self, kind, detail=""):
        line = json.dumps(
            {"ts": time.time(), "kind": kind, "detail": detail},
            ensure_ascii=False,
        )
        self._events.write(line + "\n")
        print(f"[session] {kind}: {detail}", flush=True)

    def sound(self, kind):
        if self.robot is None:
            self.event("sound_skipped", f"{kind} (no robot link)")
            return
        try:
            self.robot.signal(kind)
            self.event("sound", kind)
        except Exception as exc:
            self.event("sound_failed", f"{kind}: {exc}")

    def request_stop(self, reason):
        if self._stop.is_set():
            return
        self._stop_reason = reason
        self.collector.stop_requested = True
        self.collector.stop_reason = reason
        self._stop.set()
        self.event("stop_requested", reason)


def wake_robot_wifi(ip, rounds=4):
    """The Z10 Pro sleeps Wi-Fi aggressively while idle; a short ping
    burst wakes it (replies come back bursting: seconds -> ~1 ms)."""
    import subprocess
    import sys as _sys

    count_flag = "-n" if _sys.platform == "win32" else "-c"
    try:
        subprocess.run(
            ["ping", count_flag, "4", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        pass


def connect_with_wake_retries(session, attempts=3):
    """Connect to the Dreame, waking its Wi-Fi between attempts."""
    import os as _os

    from .robot_link import ROBOT_IP, RobotLink

    ip = _os.getenv("DREAME_IP", ROBOT_IP)
    token = _os.getenv("DREAME_TOKEN")
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            robot = RobotLink(token)
            robot.start()
            session.event("robot_connected", f"ip={ip} attempt={attempt}")
            return robot
        except Exception as exc:
            last_exc = exc
            session.event("robot_connect_failed", f"attempt={attempt}: {exc}")
            if attempt < attempts:
                session.event("robot_wake", f"ping burst to {ip}")
                wake_robot_wifi(ip)
                time.sleep(5)
    session.event("robot_unreachable", f"after {attempts} attempts: {last_exc}")
    return None


# Nominal loop rate the motion speeds were tuned against. Measured
# phone inference FPS below this scales speeds down proportionally
# (never up): the robot must not outrun fresh frames.
NOMINAL_FPS = 8.0
MIN_SPEED_SCALE = 0.4


def speed_scale_for_fps(fps):
    if fps <= 0:
        return MIN_SPEED_SCALE
    return max(MIN_SPEED_SCALE, min(1.0, fps / NOMINAL_FPS))


def write_status(session, args, started_at, deadline, now):
    """STATUS.md: glanceable progress for a human away from the PC."""
    hb_age = now - session.collector.last_heartbeat_at if session.collector.last_heartbeat_at else -1
    if session.robot is not None:
        rtt_ms, failures = session.robot.snapshot()
        vel, rot, src = session.robot.command.get_for_send()
        robot_line = (
            f"robot: cmd v={vel} r={rot} ({src}), rtt={rtt_ms:.0f}ms, "
            f"failures={failures}, patrol_scale={session.patrol_scale:.2f}"
        )
    else:
        robot_line = "robot: observe-only (no link)"
    lines = [
        "# Cat Stalker session (live)",
        "",
        f"mode={args.mode} stop={session.collector.stop_requested} "
        f"reason={session.collector.stop_reason}",
        f"elapsed={(now - started_at) / 60:.1f} min, "
        f"remaining={max(0.0, (deadline - now)) / 60:.1f} min",
        f"phone: rows={session.collector.rows_received} "
        f"frames={session.collector.frames_received} "
        f"heartbeat={'never' if hb_age < 0 else f'{hb_age:.0f}s ago'} "
        f"inference_fps={session.collector.last_inference_fps:.1f}",
        robot_line,
        "",
        "Watch the speaker: locate = come here, 1x play = done, 2x play = error.",
    ]
    try:
        with open(os.path.join(session.dir, "STATUS.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def analyzer_loop(session, every_min):
    """Periodic in-session analysis. Never touches models or devices."""
    from . import analyze as _analyze

    interval = every_min * 60
    # First run happens after one interval so data can accumulate.
    while not session._stop.wait(interval):
        try:
            report = _analyze.analyze_session(session.dir)
            session.event(
                "analysis",
                f"rows={report['telemetry']['rows']} "
                f"det_rate={report['telemetry'].get('detection_rate', 0):.1%} "
                f"frames={report['frames_kept']}/{report['frames_total']} "
                f"borderline={len(report['borderline'])} :: {report['decision']}",
            )
        except Exception as exc:
            session.event("analysis_failed", str(exc))


SEARCH_ROTATION = 14
SEARCH_BUDGET_S = 20.0
# LOST_WAIT holds still this long before sweeping: detector flicker
# (single ~1 s gaps inside real streaks) must read as a pause, not a
# reason to spin. Real losses still sweep after the hold expires.
LOST_HOLD_S = 1.5
# Sweep only if the target has been unseen for this long. At ~1.5 m a
# sitting cat flickers with multi-second gaps; each gap used to spin
# the robot away into a side orbit. Recency (not area) decides: seen
# <6 s ago means "right here, wait", unseen longer means "search".
SWEEP_GRACE_S = 6.0
# Slew limits per follow-loop cycle (8 Hz): motion commands ramp instead
# of stepping 0<->6<->10<->14 on every detector flicker. Full swing in
# ~0.4 s: smooths spin-up/down without hurting tracking. Stops, holds
# and giveups bypass the slew and go direct.
SLEW_ROTATION = 5
SLEW_VELOCITY = 8  # was 4; keep up with the 2x cruise (full swing ~0.6 s)


def _slew(current, target, step):
    delta = target - current
    if abs(delta) <= step:
        return target
    return current + step if delta > 0 else current - step
# 2026-10-06: 30 s of sitting still between sweeps reads as "dead".
# 10 s still lets a walking cat enter the frame.
SEARCH_LISTEN_S = 10.0
FOLLOW_HZ = 8.0


def follow_loop(session, robot):
    """Closed-loop follow on streamed phone telemetry.

    TRACKING -> steer + approach. Fresh loss -> stop, keep lock.
    LOST_WAIT -> bounded in-place search sweep, then stop + attention
    and back to SEARCHING. Stale frames always stop.
    """
    from . import follow as _follow

    lock = _follow.TargetLock()
    steering = _follow.SteeringController()
    forward = _follow.ForwardController()
    last_reported = ""
    searching_since = 0.0
    last_sweep_end = 0.0
    search_announced = False
    search_dir = 1
    fault_latched = False
    last_v = 0
    last_r = 0
    last_track_s = 0.0

    def sweep_motion(now):
        # Continuous one-direction sweep. The old +/- flip every few
        # seconds netted ~zero: the robot paced inside one sector
        # ("stuck angle") and never completed a full turn. A 20 s
        # budget at r=10 must cover a full 360.
        return search_dir * SEARCH_ROTATION

    while not session._stop.is_set():
        loop_start = time.monotonic()
        now_wall = time.time()
        row = dict(session.collector.last_row)
        row_at = session.collector.last_row_at
        age = now_wall - row_at if row_at else None

        detected = row.get("state") == "DETECTED"
        try:
            cx = float(row.get("offset_x")) if detected else 0.0
            area = float(row.get("area_ratio")) if detected else 0.0
            conf = float(row.get("confidence")) if detected else 0.0
        except (TypeError, ValueError):
            detected, cx, area, conf = False, 0.0, 0.0, 0.0

        snap = lock.update(detected, cx, area, conf)
        state = snap["state"]

        if session._fault_held:
            if not fault_latched:
                # A fault means a human (or a cliff edge) moved the
                # robot: the old lock episode is stale. Reset so the
                # robot re-acquires fresh instead of resuming a sweep
                # that faces wherever it was carried to.
                fault_latched = True
                lock.reset()
                steering.reset()
                forward.reset()
                session.event("follow", "FAULT_HOLD lock reset, re-acquire after clear")
            robot.command.set(0, 0, "FAULT_HOLD")
            last_v, last_r = 0, 0
            summary = f"{state} FAULT_HOLD"
            if summary != last_reported:
                last_reported = summary
                session.event("follow", summary)
            elapsed = time.monotonic() - loop_start
            session._stop.wait(max(0.0, 1.0 / FOLLOW_HZ - elapsed))
            continue
        fault_latched = False

        if state == "LOST_WAIT":
            # Fresh loss: start (or continue) a sweep episode, first
            # toward the side the target exited on; with no exit side
            # alternate per episode for full-room coverage.
            if not searching_since:
                searching_since = loop_start
                if lock.exit_sign != 0:
                    search_dir = lock.exit_sign
                else:
                    search_dir = -search_dir
                session.event("search_start", "target lost, sweeping in place")
            if loop_start - searching_since > SEARCH_BUDGET_S:
                robot.command.set(0, 0, "SEARCH_GIVEUP")
                last_v, last_r = 0, 0
                lock.reset()
                searching_since = 0.0
                last_sweep_end = loop_start
                session.event("search_giveup", "budget spent, listening")
                if not search_announced:
                    search_announced = True
                    session.sound(SIG_ATTENTION)
            elif (
                loop_start - searching_since < LOST_HOLD_S
                or snap["area_ratio"] >= _follow.CLOSE_HOLD_AREA
                or loop_start - last_track_s < SWEEP_GRACE_S
            ):
                # Hold position: fresh flicker gap (pause, not a spin),
                # lost while CLOSE, or seen seconds ago (a sitting cat at
                # ~1.5 m flickers with multi-second gaps; sweeping then
                # orbits away from an adjacent target). Sweep only for a
                # truly stale loss once all three expire.
                robot.command.set(0, 0, "CLOSE_HOLD")
                last_v, last_r = 0, 0
            else:
                rotation = _slew(last_r, sweep_motion(loop_start), SLEW_ROTATION)
                last_v, last_r = 0, rotation
                robot.command.set(0, rotation, "SEARCH")
        elif state == "SEARCHING" and (
            last_sweep_end == 0
            or loop_start - last_sweep_end > SEARCH_LISTEN_S
        ):
            # No target in view: sweep on start, then listen/sweep
            # cycles until the session ends or the cat is acquired.
            if not searching_since:
                searching_since = loop_start
                if lock.exit_sign != 0:
                    search_dir = lock.exit_sign
                # Else: keep the previous episode's direction. Alternating
                # back retraces the same sector (2026-10-06: ~180 deg per
                # episode at r=10, robot paced forth-and-back while the cat
                # sat behind it in the never-scanned half). Same-direction
                # episodes accumulate coverage to a full 360+.
                session.event(
                    "search_retry" if last_sweep_end else "search_start",
                    "no target in view, sweeping in place",
                )
            if searching_since and loop_start - searching_since > SEARCH_BUDGET_S:
                robot.command.set(0, 0, "SEARCH_GIVEUP")
                last_v, last_r = 0, 0
                searching_since = 0.0
                last_sweep_end = loop_start
                session.event("search_giveup", "budget spent, listening")
                if not search_announced:
                    search_announced = True
                    session.sound(SIG_ATTENTION)
            else:
                rotation = _slew(last_r, sweep_motion(loop_start), SLEW_ROTATION)
                last_v, last_r = 0, rotation
                robot.command.set(0, rotation, "SEARCH")
        else:
            searching_since = 0.0
            if state == "TRACKING":
                last_track_s = loop_start
            velocity, rotation, source = _follow.decide(
                snap, steering, forward, age
            )
            scale = session.patrol_scale
            velocity = int(round(velocity * scale))
            rotation = int(round(rotation * scale))
            if source in ("AUTOPILOT", "SEARCH"):
                velocity = _slew(last_v, velocity, SLEW_VELOCITY)
                rotation = _slew(last_r, rotation, SLEW_ROTATION)
            last_v, last_r = velocity, rotation
            robot.command.set(velocity, rotation, source)

        summary = f"{state} v={robot.command.velocity} r={robot.command.rotation}"
        if summary != last_reported:
            last_reported = summary
            session.event("follow", summary)

        elapsed = time.monotonic() - loop_start
        session._stop.wait(max(0.0, 1.0 / FOLLOW_HZ - elapsed))
    robot.command.set(0, 0, "FOLLOW_END")


def patrol_loop(session, robot):
    # NOTE: the motion slot expires non-zero commands after 0.5 s
    # (WATCHDOG). Patrol must re-assert continuously, otherwise the
    # robot gets a single rotation blip followed by (0,0) keepalive
    # and then does whatever its own obstacle logic dictates.
    direction = 1
    while not session._stop.is_set():
        interval_end = time.monotonic() + session.args.patrol_interval
        while not session._stop.is_set() and time.monotonic() < interval_end:
            if session._fault_held:
                robot.command.set(0, 0, "FAULT_HOLD")
            elif session._phone_held:
                robot.command.set(0, 0, "PHONE_SILENT_HOLD")
            else:
                rotation = round(session.args.patrol_rotation * session.patrol_scale)
                robot.command.set(0, direction * rotation, "PATROL")
            if session._stop.wait(0.4):
                break
        direction *= -1
    robot.command.set(0, 0, "PATROL_END")


def main(argv=None):
    args = parse_args(argv)
    session = Session(args)

    token = os.getenv("DREAME_TOKEN")
    if token:
        session.robot = connect_with_wake_retries(session)
    else:
        print(
            "[session] DREAME_TOKEN is not set: observe-only mode. "
            "Phone telemetry is still collected, but there is no robot "
            "motion, keepalive, or speaker signaling.",
            flush=True,
        )
        session.event("observe_only", "DREAME_TOKEN missing")

    import threading as _t

    server_thread = _t.Thread(
        target=session.collector.serve, args=(args.host, args.port), daemon=True
    )
    server_thread.start()

    discovery = DiscoveryResponder(args.port)
    discovery.start()
    session.event("discovery_listening", f"udp={protocol.DISCOVERY_PORT}")

    analyzer_thread = None
    if args.analyze_every_min > 0:
        analyzer_thread = _t.Thread(
            target=analyzer_loop, args=(session, args.analyze_every_min), daemon=True
        )
        analyzer_thread.start()
        session.event("analyzer_on", f"every {args.analyze_every_min} min")

    patrol_thread = None
    if args.mode == "patrol" and session.robot is not None:
        patrol_thread = _t.Thread(
            target=patrol_loop, args=(session, session.robot), daemon=True
        )
        patrol_thread.start()
    elif args.mode == "patrol":
        session.event("patrol_disabled", "no robot link; falling back to observe")

    follow_thread = None
    if args.mode == "follow" and session.robot is not None:
        follow_thread = _t.Thread(
            target=follow_loop, args=(session, session.robot), daemon=True
        )
        follow_thread.start()
    elif args.mode == "follow":
        session.event("follow_disabled", "no robot link; falling back to observe")

    deadline = time.time() + args.duration_min * 60
    started_at = time.time()
    last_status_write = 0.0
    if args.mode in ("patrol", "follow"):
        print(
            "[session] MOUNT CHECKLIST (motion mode): phone LOW at the front, "
            "LANDSCAPE (portrait mount is unverified: the steering axis "
            "mapping was validated for landscape only), "
            "must NOT cover the top lidar turret or cliff sensors; robot "
            "starts >1 m from walls/dock; human in the same room; "
            "dustbin empty. Ctrl+C = immediate safe stop.",
            flush=True,
        )
        session.event("mount_checklist_shown", args.mode)
    print(
        f"[session] dir={session.dir} mode={args.mode} "
        f"phone_url=http://{lan_ip()}:{args.port} "
        f"(enter host {lan_ip()} port {args.port} in the app Session row, then Start)",
        flush=True,
    )
    session.event("started", f"mode={args.mode}")

    try:
        while not session._stop.is_set():
            time.sleep(1.0)
            now = time.time()

            # Robot telemetry row (or observe-only marker).
            if session.robot is not None:
                rtt_ms, failures = session.robot.snapshot()
                vel, rot, src = session.robot.command.get_for_send()
                session._robot_writer.writerow(
                    [f"{now:.3f}", vel, rot, src, f"{rtt_ms:.1f}", failures]
                )
                # Device-status interlock (~every 5 s): a nonzero fault
                # (blocked laser, stuck bumper, ...) holds all motion.
                if int(now) % 5 == 0:
                    info = session.robot.poll_status()
                    if info:
                        session._status_writer.writerow([
                            f"{now:.3f}",
                            info.get("battery_level", ""),
                            info.get("device_fault", ""),
                            info.get("device_status", ""),
                        ])
                        if info.get("device_fault"):
                            if not session._fault_held:
                                session._fault_held = True
                                session.robot.command.set(0, 0, "FAULT_HOLD")
                                session.event(
                                    "fault_hold",
                                    f"device_fault={info.get('device_fault')} "
                                    f"status={info.get('device_status')}",
                                )
                                session.sound(SIG_ATTENTION)
                        elif session._fault_held:
                            session._fault_held = False
                            session.event("fault_cleared", "motion allowed again")
                if session.robot.error is not None:
                    session.event("robot_error", str(session.robot.error))
                    session.sound(SIG_ERROR)
                    session.request_stop(f"robot_error: {session.robot.error}")
                    break
                # Link-degraded: persistent failures -> halt patrol, attention.
                if failures > 0:
                    if not session._link_degraded_since:
                        session._link_degraded_since = now
                    elif now - session._link_degraded_since > 10:
                        session.robot.command.set(0, 0, "LINK_DEGRADED")
                        if not session._attention_sounded:
                            session.sound(SIG_ATTENTION)
                            session._attention_sounded = True
                else:
                    session._link_degraded_since = 0.0
            else:
                session._robot_writer.writerow(
                    [f"{now:.3f}", "", "", "OBSERVE_ONLY", "", ""]
                )

            # Phone heartbeat watchdog. Silence first holds motion
            # (keepalive only), then stops the session entirely.
            last_hb = session.collector.last_heartbeat_at
            if last_hb > 0:
                age = now - last_hb
                if age <= protocol.PHONE_HEARTBEAT_GRACE_S:
                    session._phone_held = False
                if age > protocol.PHONE_HEARTBEAT_FATAL_S:
                    session.event("phone_silent_fatal", f"{age:.0f}s without heartbeat")
                    session.sound(SIG_ERROR)
                    session.request_stop("phone silent >60s")
                    break
                if age > protocol.PHONE_HEARTBEAT_GRACE_S:
                    if session.robot is not None and not session._phone_held:
                        session.robot.command.set(0, 0, "PHONE_SILENT_HOLD")
                        session._phone_held = True
                    if not session._attention_sounded:
                        session.event("phone_silent", f"{age:.0f}s without heartbeat")
                        session.sound(SIG_ATTENTION)
                        session._attention_sounded = True
            else:
                age = None

            # Adaptive speed governor: never outrun fresh frames.
            fps = session.collector.fresh_inference_fps()
            if fps > 0:
                session.patrol_scale = speed_scale_for_fps(fps)
                session._phone_held = False
            elif last_hb > 0:
                # Stale FPS but live heartbeat: hold last safe scale.
                pass
            else:
                session.patrol_scale = MIN_SPEED_SCALE

            # Human-readable live progress (dashboard companion).
            if now - last_status_write >= 5:
                last_status_write = now
                write_status(session, args, started_at, deadline, now)

            # Duration + disk watchdogs.
            if now >= deadline:
                session.event("duration_reached", f"{args.duration_min} min")
                session.sound(SIG_DONE)
                session.request_stop("duration reached")
                break
            try:
                free = shutil.disk_usage(session.dir).free
                if free < 1024 * 1024 * 1024:
                    session.event("disk_low", f"{free} bytes free")
                    session.sound(SIG_ERROR)
                    session.request_stop("disk <1GB")
                    break
            except OSError:
                pass
    except KeyboardInterrupt:
        session.event("interrupted", "Ctrl+C: safe stop")
        session.sound(SIG_DONE)
        session.request_stop("interrupted")
    finally:
        session.collector.stop_requested = True
        discovery.shutdown()
        if analyzer_thread is not None:
            analyzer_thread.join(timeout=5)
        if follow_thread is not None:
            follow_thread.join(timeout=5)
        if patrol_thread is not None:
            patrol_thread.join(timeout=5)
        if session.robot is not None:
            session.robot.shutdown()
        session.collector.shutdown()
        session.event(
            "finished",
            f"reason={session._stop_reason} "
            f"phone_rows={session.collector.rows_received} "
            f"frames={session.collector.frames_received}",
        )
        try:
            session._robot_csv.close()
            session._status_csv.close()
            session._events.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
