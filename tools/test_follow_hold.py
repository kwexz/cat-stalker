"""Harness: follow_loop must HOLD (not sweep) shortly after losing a
fresh target, and sweep only once the loss is truly stale.

Run: python tools/test_follow_hold.py
"""

import sys
import threading
import time

sys.path.insert(0, ".")
from session import auto_session as A


class FakeCmd:
    def __init__(self):
        self.log = []

    def set(self, v, r, s):
        self.log.append((v, r, s))

    @property
    def velocity(self):
        return self.log[-1][0] if self.log else 0

    @property
    def rotation(self):
        return self.log[-1][1] if self.log else 0


class FakeCol:
    last_row = {}
    last_row_at = 0.0


class FakeRobot:
    def __init__(self):
        self.command = FakeCmd()


class FakeSess:
    def __init__(self):
        self._stop = threading.Event()
        self._fault_held = False
        self.patrol_scale = 1.0
        self.collector = FakeCol()

    def event(self, kind, detail=""):
        pass

    def sound(self, kind):
        pass


def main():
    A.SWEEP_GRACE_S = 6.0
    A.LOST_HOLD_S = 1.5
    sess = FakeSess()
    robot = FakeRobot()
    t = threading.Thread(target=A.follow_loop, args=(sess, robot), daemon=True)
    t.start()

    def feed(detected):
        if detected:
            sess.collector.last_row = {
                "state": "DETECTED",
                "offset_x": "-0.1",
                "area_ratio": "0.05",
                "confidence": "0.6",
            }
            sess.collector.last_row_at = time.time()
        else:
            sess.collector.last_row = {}
            sess.collector.last_row_at = 0.0

    # 1 s of solid tracking (far-ish area 0.05: below CLOSE_HOLD_AREA, so
    # only the recency grace can suppress the sweep).
    for _ in range(10):
        feed(True)
        time.sleep(0.1)
    assert any(s == "AUTOPILOT" for _, _, s in robot.command.log), "never tracked"
    robot.command.log.clear()

    # Silence: TEMP_LOST coast, then hold. No SEARCH for SWEEP_GRACE_S.
    feed(False)
    time.sleep(4.0)
    sources = {s for _, _, s in robot.command.log}
    print("sources during 4 s silence:", sorted(sources))
    assert "SEARCH" not in sources, f"swept a 4 s-old loss: {sources}"
    assert "CLOSE_HOLD" in sources, "expected hold, got nothing"

    # Keep silent past the grace: sweep must appear.
    time.sleep(5.0)
    sources = {s for _, _, s in robot.command.log}
    print("sources during 9 s silence:", sorted(sources))
    assert "SEARCH" in sources, "never swept a stale loss"

    sess._stop.set()
    t.join(timeout=3)
    print("HOLD-THEN-SWEEP OK")


if __name__ == "__main__":
    main()
