"""UDP discovery responder: lets the P30 find the PC with zero taps.

Phone broadcasts protocol.DISCOVER_MSG on protocol.DISCOVERY_PORT;
we reply with our HTTP port. Runs in its own daemon thread.
"""

import socket
import threading

from . import protocol


class DiscoveryResponder:
    def __init__(self, http_port):
        self.http_port = http_port
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except OSError:
                pass
            sock.bind(("0.0.0.0", protocol.DISCOVERY_PORT))
            sock.settimeout(1.0)
            reply = protocol.REPLY_PREFIX + str(self.http_port).encode("ascii")
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(256)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if data == protocol.DISCOVER_MSG:
                    try:
                        sock.sendto(reply, addr)
                    except OSError:
                        pass
        finally:
            sock.close()

    def shutdown(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
