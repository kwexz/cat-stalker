"""Minimal Dreame Z10 Pro link for autonomous sessions.

Duplicates only the proven mapping from the Windows prototype
(`cat_stalker.py`, which must stay untouched):
- move: siid 21 / aiid 1, piid 1 = rotation, piid 2 = velocity
- locate: siid 7 / aiid 1, play_sound: siid 7 / aiid 2
- idle keepalive via move(0, 0) at 1.5 Hz (read-only heartbeat does
  NOT keep the Z10 Pro Wi-Fi awake), moving send rate 8 Hz
- stale non-zero commands expire after 0.5 s (WATCHDOG -> stop)

Sounds are the human-attention channel: the human stays in the
apartment but not at the PC.
"""

import os
import threading
import time

ROBOT_IP = os.getenv("DREAME_IP", "192.168.1.124")
ROBOT_MODEL = "dreame.vacuum.p2028"

MOVING_HZ = 8.0
IDLE_KEEPALIVE_HZ = 1.5
COMMAND_TIMEOUT_S = 0.5
MAX_FAILURES = 5

# Sound-signal vocabulary (all via the robot speaker).
SIG_ATTENTION = "locate"      # human needed at the robot/phone
SIG_DONE = "play_once"        # session finished, come collect
SIG_ERROR = "play_twice"      # error stop, human needed


class LatestCommand:
    """Single-slot desired motion. Never queues stale commands."""

    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.velocity = 0
        self.rotation = 0
        self.source = "IDLE"
        self.updated_at = time.monotonic()

    def set(self, velocity, rotation, source):
        with self._lock:
            self.updated_at = time.monotonic()
            if (self.velocity, self.rotation, self.source) != (
                int(velocity), int(rotation), source,
            ):
                self.velocity = int(velocity)
                self.rotation = int(rotation)
                self.source = source
                self._event.set()

    def get_for_send(self):
        with self._lock:
            now = time.monotonic()
            if (
                (self.velocity != 0 or self.rotation != 0)
                and now - self.updated_at > COMMAND_TIMEOUT_S
            ):
                return 0, 0, "WATCHDOG"
            return self.velocity, self.rotation, self.source

    def wait(self, timeout):
        hit = self._event.wait(timeout)
        self._event.clear()
        return hit

    def wake(self):
        self._event.set()


class RobotLink:
    def __init__(self, token):
        from miio.integrations.dreame.vacuum.dreamevacuum_miot import (
            DreameVacuum,
        )

        self.robot = DreameVacuum(ROBOT_IP, token, model=ROBOT_MODEL, timeout=2)
        self.command = LatestCommand()
        self._stop = threading.Event()
        self._thread = None
        self.last_rtt_ms = 0.0
        self.failures = 0
        self.error = None
        self._lock = threading.Lock()

    @staticmethod
    def _move_params(velocity, rotation):
        return [
            {"piid": 1, "value": str(rotation)},
            {"piid": 2, "value": str(velocity)},
        ]

    def _send_move(self, velocity, rotation):
        started = time.perf_counter()
        self.robot.call_action_from_mapping(
            "move", self._move_params(velocity, rotation)
        )
        with self._lock:
            self.last_rtt_ms = (time.perf_counter() - started) * 1000.0
            self.failures = 0

    def snapshot(self):
        with self._lock:
            return self.last_rtt_ms, self.failures

    def poll_status(self):
        """Best-effort device status. Never raises; returns {} on failure.

        Keys: battery_level, device_fault (int, 0 == ok), device_status.
        A nonzero fault (e.g. blocked laser sensor) means the robot's
        own safety logic is active and our motion must hold.
        """
        try:
            status = self.robot.status()
            data = status.data if hasattr(status, "data") else {}
            fault = data.get("device_fault", 0) or 0
            return {
                "battery_level": data.get("battery_level"),
                "device_fault": int(fault),
                "device_status": data.get("device_status"),
            }
        except Exception:
            return {}

    def start(self):
        self._send_move(0, 0)
        time.sleep(0.15)
        try:
            self.robot.set_fan_speed(0)  # Quiet; motor still runs in manual mode
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        last_send = 0.0
        try:
            while not self._stop.is_set():
                velocity, rotation, _ = self.command.get_for_send()
                interval = (
                    1.0 / MOVING_HZ
                    if (velocity or rotation)
                    else 1.0 / IDLE_KEEPALIVE_HZ
                )
                self.command.wait(max(0.0, interval - (time.monotonic() - last_send)))
                velocity, rotation, _ = self.command.get_for_send()
                try:
                    self._send_move(velocity, rotation)
                    last_send = time.monotonic()
                except Exception as exc:  # DeviceException/OSError/TimeoutError
                    with self._lock:
                        self.failures += 1
                        if self.failures >= MAX_FAILURES:
                            self.error = exc
                            return
                    time.sleep(0.05)
        finally:
            try:
                self._send_move(0, 0)
            except Exception:
                pass
            try:
                self.robot.call_action_from_mapping("stop_clean", [])
            except Exception:
                pass

    # -- speaker signals ------------------------------------------------
    def _action(self, name):
        self.robot.call_action_from_mapping(name, [])

    def locate(self):
        self._action("locate")

    def play_sound(self):
        self._action("play_sound")

    def signal(self, kind):
        if kind == SIG_ATTENTION:
            self.locate()
        elif kind == SIG_DONE:
            self.play_sound()
        elif kind == SIG_ERROR:
            self.play_sound()
            time.sleep(1.2)
            self.play_sound()
        else:
            raise ValueError(f"unknown signal: {kind}")

    def shutdown(self):
        self.command.set(0, 0, "SHUTDOWN")
        self._stop.set()
        self.command.wake()
        if self._thread is not None:
            self._thread.join(timeout=4)
