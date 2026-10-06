"""Visual-servo following on streamed phone telemetry.

Ports the proven math from the Windows prototype (`cat_stalker.py`):
stable target lock with TEMP_LOST/LOST hysteresis, P+D steering with
center deadband + brake-on-crossing, forward motion gated by apparent
area with start/stop hysteresis.

Differences from the prototype (deliberate, documented):
- No ByteTrack IDs on the phone yet: the stream carries one best
  detection per frame, so the lock is persistence + EMA smoothing
  instead of a track-ID lock. A wrong-target switch is possible if
  two cats alternate frames; the single-cat apartment is assumed.
- Speeds are capped BELOW the prototype values for first robot runs.
- Area thresholds are copied from the prototype but marked
  UNCALIBRATED for the low (~10 cm) robot camera viewpoint. They must
  be re-measured from the robot before trusting approach distance.
- Forward motion additionally requires fresh frames (age <= 0.5 s)
  and live speed-governor scale from the session.
"""

import math
import time

# --- acquisition / loss timing ---
# Prototype wanted 5 CONSECUTIVE frames, but the phone detector flickers
# (single missed frames inside real streaks), so consecutive counting
# almost never acquires. M-of-N window instead: 3 of last 5.
LOCK_WINDOW = 5
LOCK_REQUIRED_HITS = 3
TEMP_LOST_S = 0.6
LOST_S = 1.5

# --- smoothing ---
EMA_ALPHA_POSITION = 0.60
EMA_ALPHA_SIZE = 0.25

# --- steering (P + small D, hysteresis, brake) ---
CENTER_INNER = 0.05
CENTER_OUTER = 0.10
KP_ROTATION = 24.0
KD_ROTATION = 3.0
# 2026-10-06: 75 s of r=+6/+7 held offset frozen at -0.2..-0.3 (session
# 20261006_211313) while |r|>=14 visibly turns. Small commands sit in a
# firmware deadzone/stiction band, so MIN 6 meant "track but never
# center" (forward crawled uncentered = "approach from the side", only
# SEARCH sweeps turned). 10 is above the deadzone; watch for
# center overshoot oscillation, then tune the deadband instead.
MIN_AUTO_ROTATION = 10
MAX_AUTO_ROTATION = 22  # prototype value; 2026-10-06: 18 still cannot
# keep up with a walking cat (~9 deg/s at r=14 vs ~28 deg/s for a cat
# walking tangentially at 2 m). Rotation-only chase of a moving cat
# always lags; this just shortens the lag. Real fix is catching up
# while the cat sits still.

# --- forward (apparent-area hysteresis) ---
# UNCALIBRATED for the low robot camera. Remeasure area at the desired
# real follow distance from the mounted phone, then adjust.
FORWARD_START_AREA = 0.095
FORWARD_STOP_AREA = 0.135
# 0.65: the 2026-10-06 cat sat at |offset|~0.58, i.e. left third of
# the wide phone frame, not the extreme edge. At full turn the speed
# is already scaled to a creep (v=4), so blocking below the last
# frame sliver only.
FORWARD_MAX_OFFSET = 0.65
# Lowered from the prototype's 0.38: P30 detections live in 0.3-0.5,
# and the gate blocked approach on most real streaks.
FORWARD_MIN_CONFIDENCE = 0.32
# Hysteresis against single-frame conf dips at close range (2026-10-06
# pillow run: 13% of close rows sat in 0.28-0.31 and each dip dropped
# the forward latch, flickering v=0/6). Start at 0.32, stop at 0.25.
FORWARD_STOP_CONFIDENCE = 0.25
# Close-loss memory: LOST while the smoothed area was at/above this
# means the target is adjacent (far field sits <=0.04, approach rows
# run 0.07+). Sweeping away from an adjacent target is always wrong,
# so the loop holds position instead. Below FORWARD_START_AREA on
# purpose: the EMA lags sudden growth at close range.
CLOSE_HOLD_AREA = 0.06
MAX_AUTO_FORWARD = 40  # 2026-10-06: 10 crawled, 20 still slow; 2x again per user

# --- freshness ---
MAX_FRAME_AGE_S = 0.5


def _clamp(value, low, high):
    return max(low, min(high, value))


def _ema(previous, current, alpha):
    if previous is None:
        return current
    return previous * (1.0 - alpha) + current * alpha


class TargetLock:
    """SEARCHING -> TRACKING -> TEMP_LOST -> LOST_WAIT -> SEARCHING."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "SEARCHING"
        from collections import deque

        self._window = deque(maxlen=LOCK_WINDOW)
        self._last_seen = 0.0
        self.smooth_cx = None
        self.smooth_area = None
        self.last_conf = 0.0
        # Side the target was last seen on (+1 right, -1 left, 0
        # unknown): the search sweep starts toward the exit side.
        self.exit_sign = 0

    def update(self, detected, cx_norm, area, conf, now=None):
        """cx_norm: target center x in [-1, 1] frame-normalized coords."""
        now = now if now is not None else time.monotonic()
        if detected:
            if self.state in ("SEARCHING", "LOST_WAIT"):
                self._window.append(True)
                if sum(self._window) >= LOCK_REQUIRED_HITS:
                    self.state = "TRACKING"
                    self.smooth_cx = None
                    self.smooth_area = None
                    self._window.clear()
                else:
                    return self._snapshot()
            else:
                self._window.clear()
            self.state = "TRACKING"
            self._last_seen = now
            if cx_norm > 0:
                self.exit_sign = 1
            elif cx_norm < 0:
                self.exit_sign = -1
            self._candidate_frames = 0
            self.smooth_cx = _ema(self.smooth_cx, cx_norm, EMA_ALPHA_POSITION)
            self.smooth_area = _ema(self.smooth_area, area, EMA_ALPHA_SIZE)
            self.last_conf = conf
            return self._snapshot()
        # No detection this frame.
        if self.state in ("SEARCHING", "LOST_WAIT"):
            self._window.append(False)
            return self._snapshot()
        missing = now - self._last_seen
        if missing <= TEMP_LOST_S:
            self.state = "TEMP_LOST"
        elif missing <= LOST_S:
            self.state = "LOST_WAIT"
        else:
            self.reset()
        return self._snapshot()

    def _snapshot(self):
        offset_x = None
        if self.smooth_cx is not None:
            offset_x = self.smooth_cx
        return {
            "state": self.state,
            "offset_x": offset_x,
            "area_ratio": self.smooth_area or 0.0,
            "confidence": self.last_conf,
        }


class SteeringController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turning = False
        self.turn_sign = 0
        self.last_error = None
        self.last_time = None

    def update(self, error, now=None):
        now = now if now is not None else time.monotonic()
        if error is None:
            self.reset()
            return 0
        sign = 1 if error > 0 else (-1 if error < 0 else 0)
        derivative = 0.0
        if self.last_error is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 1e-3:
                derivative = (error - self.last_error) / dt
        self.last_error = error
        self.last_time = now
        if self.turning and sign != 0 and self.turn_sign != 0 and sign != self.turn_sign:
            self.turning = False
            self.turn_sign = 0
            return 0
        if self.turning:
            if abs(error) <= CENTER_INNER:
                self.turning = False
                self.turn_sign = 0
                return 0
        elif abs(error) < CENTER_OUTER:
            return 0
        else:
            self.turning = True
            self.turn_sign = sign
        rotation = -KP_ROTATION * error - KD_ROTATION * derivative
        rotation = _clamp(rotation, -MAX_AUTO_ROTATION, MAX_AUTO_ROTATION)
        if 0 < abs(rotation) < MIN_AUTO_ROTATION:
            rotation = math.copysign(MIN_AUTO_ROTATION, rotation)
        return int(round(rotation))


class ForwardController:
    def __init__(self):
        self.moving = False

    def reset(self):
        self.moving = False

    def update(self, target, steering_rotation):
        # TEMP_LOST coasts like TRACKING (frozen smoothed target, see
        # decide()): single flickered frames must not stutter the drive.
        if target is None or target["state"] not in ("TRACKING", "TEMP_LOST"):
            self.moving = False
            return 0
        offset = target["offset_x"]
        area = target["area_ratio"]
        conf = target["confidence"]
        if offset is None:
            self.moving = False
            return 0
        if not self.moving and conf < FORWARD_MIN_CONFIDENCE:
            return 0
        if self.moving and conf < FORWARD_STOP_CONFIDENCE:
            self.moving = False
            return 0
        if abs(offset) > FORWARD_MAX_OFFSET:
            self.moving = False
            return 0
        if self.moving:
            if area >= FORWARD_STOP_AREA:
                self.moving = False
        elif area <= FORWARD_START_AREA:
            self.moving = True
        if not self.moving:
            return 0
        turn_fraction = min(1.0, abs(steering_rotation) / max(MAX_AUTO_ROTATION, 1))
        velocity = int(round(MAX_AUTO_FORWARD * (1.0 - 0.55 * turn_fraction)))
        return max(4, velocity)


def decide(lock_snapshot, steering, forward, frame_age_s):
    """Single decision point. Stale frames always mean stop.

    TEMP_LOST coasts on the frozen smoothed target (≤0.6 s): the phone
    detector flickers single frames inside real streaks, and a full
    stop on every flicker produced drive-stop-sweep stutter. Truly
    dead streams are still caught by the frame-age gate above, and
    LOST_WAIT/SEARCHING always stop.
    """
    if frame_age_s is None or frame_age_s > MAX_FRAME_AGE_S:
        steering.reset()
        forward.reset()
        return 0, 0, "STALE_STOP"
    if lock_snapshot["state"] not in ("TRACKING", "TEMP_LOST"):
        steering.reset()
        forward.reset()
        return 0, 0, "CV_STOP"
    rotation = steering.update(lock_snapshot["offset_x"])
    velocity = forward.update(lock_snapshot, rotation)
    return velocity, rotation, "AUTOPILOT"
