import os
import sys
import time
import math
import csv
import threading
import cv2
import pygame

from ultralytics import YOLO
from miio.integrations.dreame.vacuum.dreamevacuum_miot import DreameVacuum
from miio.exceptions import DeviceException


# ============================================================
# Cat Stalker V1.6
# - faster steering response
# - slow autonomous forward motion
# - forward hysteresis based on apparent cat size
# - Xbox full manual override (steer + throttle)
# ============================================================

# ---------- Dreame ----------
ROBOT_IP = os.getenv("DREAME_IP", "192.168.1.124")
ROBOT_TOKEN = os.getenv("DREAME_TOKEN")
ROBOT_MODEL = "dreame.vacuum.p2028"

MOVING_HZ = 8.0
IDLE_KEEPALIVE_HZ = 1.5
COMMAND_TIMEOUT_SECONDS = 0.5
MAX_ROBOT_FAILURES = 5

MAX_AUTO_ROTATION = 22
MAX_AUTO_FORWARD = 10

MAX_MANUAL_ROTATION = 28
MAX_MANUAL_FORWARD = 45
MAX_MANUAL_BACKWARD = 60

# ---------- Camera / CV ----------
CAMERA_INDEX = 1
MODEL_NAME = "yolo11n.pt"
CAT_CLASS_ID = 15
CONFIDENCE = 0.28
IMAGE_SIZE = 480

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

LOCK_REQUIRED_FRAMES = 5
TEMP_LOST_SECONDS = 0.6
LOST_SECONDS = 1.5

EMA_ALPHA_POSITION = 0.60
EMA_ALPHA_SIZE = 0.25

# ---------- Steering ----------
# Faster start than v1.5.
CENTER_INNER = 0.05
CENTER_OUTER = 0.10

KP_ROTATION = 24.0
KD_ROTATION = 3.0
MIN_AUTO_ROTATION = 6

# ---------- Forward control ----------
# NOTE: these are NOT metres. They are bbox area / frame area.
#
# Start moving when cat looks smaller than this.
FORWARD_START_AREA = 0.095

# Stop moving when cat grows past this.
FORWARD_STOP_AREA = 0.135

# Never move forward unless cat is reasonably centered.
FORWARD_MAX_OFFSET = 0.13

# Extra confidence gate for movement.
FORWARD_MIN_CONFIDENCE = 0.38

# ---------- Xbox ----------
AXIS_STEER = 0
AXIS_THROTTLE = 1

BUTTON_A = 0
BUTTON_B = 1
BUTTON_Y = 3

MANUAL_OVERRIDE_THRESHOLD = 0.18
STICK_DEADZONE = 0.12

# ---------- UI ----------
WINDOW_NAME = "Cat Stalker V1.6 - follow mode"
TELEMETRY_PATH = "telemetry.csv"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ema(previous, current, alpha):
    if previous is None:
        return current
    return previous * (1.0 - alpha) + current * alpha


def apply_deadzone(value, deadzone=STICK_DEADZONE):
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def expo(value, exponent=1.6):
    return math.copysign(abs(value) ** exponent, value)


class SharedControlState:
    def __init__(self):
        self._lock = threading.Lock()

        self.autopilot = False
        self.exit_requested = False

        self.manual_override = False
        self.manual_velocity = 0
        self.manual_rotation = 0

    def snapshot(self):
        with self._lock:
            return (
                self.autopilot,
                self.exit_requested,
                self.manual_override,
                self.manual_velocity,
                self.manual_rotation,
            )

    def toggle_autopilot(self):
        with self._lock:
            self.autopilot = not self.autopilot
            return self.autopilot

    def force_stop(self):
        with self._lock:
            self.autopilot = False
            self.manual_override = False
            self.manual_velocity = 0
            self.manual_rotation = 0

    def set_manual(self, active, velocity, rotation):
        with self._lock:
            self.manual_override = active
            self.manual_velocity = int(velocity)
            self.manual_rotation = int(rotation)

            if active:
                self.autopilot = False

    def request_exit(self):
        with self._lock:
            self.exit_requested = True


class SharedCommand:
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()

        self.velocity = 0
        self.rotation = 0
        self.source = "IDLE"
        self.updated_at = time.monotonic()

    def set(self, velocity, rotation, source):
        velocity = int(velocity)
        rotation = int(rotation)

        changed = False

        with self._lock:
            self.updated_at = time.monotonic()

            if (
                self.velocity != velocity
                or self.rotation != rotation
                or self.source != source
            ):
                self.velocity = velocity
                self.rotation = rotation
                self.source = source
                changed = True

        if changed:
            self._event.set()

    def get(self):
        with self._lock:
            return self.velocity, self.rotation, self.source

    def get_for_send(self, now=None):
        with self._lock:
            if now is None:
                now = time.monotonic()
            stale = (
                (self.velocity != 0 or self.rotation != 0)
                and now - self.updated_at
                > COMMAND_TIMEOUT_SECONDS
            )
            if stale:
                return 0, 0, "WATCHDOG"
            return self.velocity, self.rotation, self.source

    def wait_for_change(self, timeout):
        changed = self._event.wait(timeout)
        self._event.clear()
        return changed

    def wake(self):
        self._event.set()


class XboxThread(threading.Thread):
    def __init__(self, state, command):
        super().__init__(daemon=True)

        self.state = state
        self.command = command

        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None

    def run(self):
        try:
            pygame.init()
            pygame.joystick.init()

            if pygame.joystick.get_count() == 0:
                raise RuntimeError("Xbox controller not detected.")

            js = pygame.joystick.Joystick(0)
            js.init()

            print(
                f"Controller: {js.get_name()} "
                f"axes={js.get_numaxes()} "
                f"buttons={js.get_numbuttons()}"
            )

            prev_y = False
            prev_a = False
            prev_b = False
            was_manual = False

            self.ready.set()

            while not self.stop_event.is_set():
                pygame.event.pump()

                y = bool(js.get_button(BUTTON_Y))
                a = bool(js.get_button(BUTTON_A))
                b = bool(js.get_button(BUTTON_B))

                if y and not prev_y:
                    enabled = self.state.toggle_autopilot()

                    self.command.set(
                        0,
                        0,
                        "AUTOPILOT_ARM"
                        if enabled
                        else "IDLE",
                    )

                    print(
                        f"\nAutopilot "
                        f"{'ON' if enabled else 'OFF'}"
                    )

                if a and not prev_a:
                    self.state.force_stop()
                    self.command.set(0, 0, "STOP")
                    print("\nSTOP")

                if b and not prev_b:
                    self.state.request_exit()
                    self.command.set(0, 0, "EXIT")

                prev_y = y
                prev_a = a
                prev_b = b

                steer = apply_deadzone(
                    js.get_axis(AXIS_STEER)
                )

                throttle = apply_deadzone(
                    -js.get_axis(AXIS_THROTTLE)
                )

                active = (
                    abs(steer) > MANUAL_OVERRIDE_THRESHOLD
                    or abs(throttle) > MANUAL_OVERRIDE_THRESHOLD
                )

                if active:
                    rotation = int(round(
                        -expo(steer)
                        * MAX_MANUAL_ROTATION
                    ))

                    if throttle >= 0:
                        velocity = int(round(
                            expo(throttle)
                            * MAX_MANUAL_FORWARD
                        ))
                    else:
                        velocity = int(round(
                            expo(throttle)
                            * MAX_MANUAL_BACKWARD
                        ))

                    self.state.set_manual(
                        True,
                        velocity,
                        rotation,
                    )

                    self.command.set(
                        velocity,
                        rotation,
                        "XBOX",
                    )
                    was_manual = True

                else:
                    self.state.set_manual(
                        False,
                        0,
                        0,
                    )

                    if was_manual:
                        self.command.set(
                            0,
                            0,
                            "XBOX_RELEASE",
                        )
                        was_manual = False

                time.sleep(0.01)

        except Exception as exc:
            self.error = exc
            self.command.set(0, 0, "XBOX_ERROR")
            self.ready.set()
            self.state.request_exit()

    def shutdown(self):
        self.stop_event.set()
        self.join(timeout=2)
        pygame.quit()


class DreameWorker(threading.Thread):
    def __init__(self, command):
        super().__init__(daemon=True)

        if not ROBOT_TOKEN:
            raise RuntimeError(
                'DREAME_TOKEN is not set. '
                'PowerShell: $env:DREAME_TOKEN="..."'
            )

        self.command = command
        self.stop_event = threading.Event()

        self.robot = DreameVacuum(
            ROBOT_IP,
            ROBOT_TOKEN,
            model=ROBOT_MODEL,
            timeout=2,
        )

        self.last_rtt_ms = 0.0
        self.failures = 0
        self.error = None

    @staticmethod
    def _move_params(velocity, rotation):
        return [
            {"piid": 1, "value": str(rotation)},
            {"piid": 2, "value": str(velocity)},
        ]

    def _send_move(self, velocity, rotation):
        started = time.perf_counter()

        self.robot.call_action_from_mapping(
            "move",
            self._move_params(
                velocity,
                rotation,
            ),
        )

        self.last_rtt_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        self.failures = 0

    def run(self):
        try:
            print(
                f"Connecting to "
                f"{ROBOT_MODEL} @ {ROBOT_IP} ..."
            )

            self._send_move(0, 0)
            time.sleep(0.15)

            try:
                self.robot.set_fan_speed(0)
                print("Fan: Quiet")
            except Exception as exc:
                print(
                    f"[WARN] Could not set Quiet: {exc}"
                )

            last_send = 0.0

            while not self.stop_event.is_set():
                velocity, rotation, _ = (
                    self.command.get_for_send()
                )

                moving = (
                    velocity != 0
                    or rotation != 0
                )

                interval = (
                    1.0 / MOVING_HZ
                    if moving
                    else 1.0 / IDLE_KEEPALIVE_HZ
                )

                elapsed = (
                    time.monotonic()
                    - last_send
                )

                timeout = max(
                    0.0,
                    interval - elapsed,
                )

                self.command.wait_for_change(
                    timeout
                )

                velocity, rotation, _ = (
                    self.command.get_for_send()
                )

                try:
                    self._send_move(
                        velocity,
                        rotation,
                    )

                    last_send = (
                        time.monotonic()
                    )

                except (
                    DeviceException,
                    OSError,
                    TimeoutError,
                ) as exc:
                    self.failures += 1

                    print(
                        f"\n[WARN] Dreame error "
                        f"{self.failures}: {exc}"
                    )

                    if self.failures >= MAX_ROBOT_FAILURES:
                        self.error = exc
                        break

                    time.sleep(0.05)

        except Exception as exc:
            self.error = exc

        finally:
            try:
                self._send_move(0, 0)
            except Exception:
                pass

            try:
                self.robot.call_action_from_mapping(
                    "stop_clean",
                    [],
                )
            except Exception:
                pass

    def shutdown(self):
        self.stop_event.set()
        self.command.wake()
        self.join(timeout=4)


class TargetTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "SEARCHING"

        self.locked_id = None
        self.candidate_id = None
        self.candidate_frames = 0

        self.last_seen_time = None

        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_area = None

        self.last_box = None
        self.last_conf = 0.0

    def _lock(self, track_id):
        self.locked_id = track_id
        self.state = "TRACKING"
        self.last_seen_time = time.monotonic()

        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_area = None

    def _candidate_update(
        self,
        detections,
    ):
        if not detections:
            self.candidate_id = None
            self.candidate_frames = 0
            return

        best = max(
            detections,
            key=lambda d: d["confidence"],
        )

        tid = best["track_id"]

        if tid == self.candidate_id:
            self.candidate_frames += 1
        else:
            self.candidate_id = tid
            self.candidate_frames = 1

        if (
            self.candidate_frames
            >= LOCK_REQUIRED_FRAMES
        ):
            self._lock(tid)

            self.candidate_id = None
            self.candidate_frames = 0

    def update(
        self,
        detections,
        frame_w,
        frame_h,
    ):
        now = time.monotonic()

        if self.locked_id is None:
            self.state = "SEARCHING"

            self._candidate_update(
                detections
            )

            if self.locked_id is None:
                return None

        target = next(
            (
                d
                for d in detections
                if d["track_id"]
                == self.locked_id
            ),
            None,
        )

        if target is not None:
            self.state = "TRACKING"
            self.last_seen_time = now

            x1, y1, x2, y2 = (
                target["box"]
            )

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            area = (
                max(0, x2 - x1)
                * max(0, y2 - y1)
                / float(
                    frame_w
                    * frame_h
                )
            )

            self.smooth_cx = ema(
                self.smooth_cx,
                cx,
                EMA_ALPHA_POSITION,
            )

            self.smooth_cy = ema(
                self.smooth_cy,
                cy,
                EMA_ALPHA_POSITION,
            )

            self.smooth_area = ema(
                self.smooth_area,
                area,
                EMA_ALPHA_SIZE,
            )

            self.last_box = target["box"]
            self.last_conf = (
                target["confidence"]
            )

            offset_x = (
                self.smooth_cx
                - frame_w / 2.0
            ) / (
                frame_w / 2.0
            )

            return {
                "state": self.state,
                "track_id": self.locked_id,
                "box": self.last_box,
                "confidence": self.last_conf,
                "center": (
                    int(self.smooth_cx),
                    int(self.smooth_cy),
                ),
                "offset_x": offset_x,
                "area_ratio": self.smooth_area,
            }

        if self.last_seen_time is None:
            self.reset()
            return None

        missing = (
            now - self.last_seen_time
        )

        if missing <= TEMP_LOST_SECONDS:
            self.state = "TEMP_LOST"
        elif missing <= LOST_SECONDS:
            self.state = "LOST_WAIT"
        else:
            self.reset()
            return None

        return {
            "state": self.state,
            "track_id": self.locked_id,
            "box": self.last_box,
            "confidence": self.last_conf,
            "center": None,
            "offset_x": None,
            "area_ratio": (
                self.smooth_area or 0.0
            ),
            "missing_for": missing,
        }


class SteeringController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turning = False
        self.turn_sign = 0

        self.last_error = None
        self.last_time = None

    def update(self, error):
        now = time.monotonic()

        if error is None:
            self.reset()
            return 0

        current_sign = (
            1 if error > 0
            else -1 if error < 0
            else 0
        )

        derivative = 0.0

        if (
            self.last_error is not None
            and self.last_time is not None
        ):
            dt = now - self.last_time

            if dt > 1e-3:
                derivative = (
                    error
                    - self.last_error
                ) / dt

        self.last_error = error
        self.last_time = now

        # Brake after crossing center,
        # instead of immediately reversing.
        if (
            self.turning
            and current_sign != 0
            and self.turn_sign != 0
            and current_sign
            != self.turn_sign
        ):
            self.turning = False
            self.turn_sign = 0
            return 0

        # Hysteresis.
        if self.turning:
            if abs(error) <= CENTER_INNER:
                self.turning = False
                self.turn_sign = 0
                return 0
        else:
            if abs(error) < CENTER_OUTER:
                return 0

            self.turning = True
            self.turn_sign = current_sign

        rotation = (
            -KP_ROTATION * error
            - KD_ROTATION * derivative
        )

        rotation = clamp(
            rotation,
            -MAX_AUTO_ROTATION,
            MAX_AUTO_ROTATION,
        )

        if (
            0
            < abs(rotation)
            < MIN_AUTO_ROTATION
        ):
            rotation = math.copysign(
                MIN_AUTO_ROTATION,
                rotation,
            )

        return int(round(rotation))


class ForwardController:
    """
    Hysteresis around apparent target size.
    Prevents forward/stop chatter around one threshold.
    """

    def __init__(self):
        self.moving = False

    def reset(self):
        self.moving = False

    def update(
        self,
        target,
        steering_rotation,
    ):
        if (
            target is None
            or target["state"]
            != "TRACKING"
        ):
            self.moving = False
            return 0

        offset = target["offset_x"]
        area = target["area_ratio"]
        confidence = target["confidence"]

        if (
            offset is None
            or confidence
            < FORWARD_MIN_CONFIDENCE
        ):
            self.moving = False
            return 0

        # Turning hard? First face the cat.
        if (
            abs(offset)
            > FORWARD_MAX_OFFSET
        ):
            self.moving = False
            return 0

        # Hysteresis on apparent distance.
        if self.moving:
            if area >= FORWARD_STOP_AREA:
                self.moving = False
        else:
            if area <= FORWARD_START_AREA:
                self.moving = True

        if not self.moving:
            return 0

        # Reduce forward speed while steering.
        turn_fraction = min(
            1.0,
            abs(steering_rotation)
            / max(MAX_AUTO_ROTATION, 1),
        )

        speed_scale = (
            1.0
            - 0.55 * turn_fraction
        )

        velocity = int(round(
            MAX_AUTO_FORWARD
            * speed_scale
        ))

        return max(4, velocity)


def extract_detections(result):
    detections = []
    boxes = result.boxes

    if (
        boxes is None
        or len(boxes) == 0
        or boxes.id is None
    ):
        return detections

    ids = (
        boxes.id
        .int()
        .cpu()
        .tolist()
    )

    confs = (
        boxes.conf
        .cpu()
        .tolist()
    )

    coords = (
        boxes.xyxy
        .cpu()
        .tolist()
    )

    for tid, conf, xyxy in zip(
        ids,
        confs,
        coords,
    ):
        detections.append({
            "track_id": int(tid),
            "confidence": float(conf),
            "box": tuple(
                map(int, xyxy)
            ),
        })

    return detections


def open_camera(index):
    cap = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}"
        )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    print(
        "Camera actual: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"FPS={cap.get(cv2.CAP_PROP_FPS):.1f}"
    )

    return cap


def main():
    print(
        f"Loading {MODEL_NAME} ..."
    )

    model = YOLO(
        MODEL_NAME
    )

    cap = open_camera(
        CAMERA_INDEX
    )

    tracker = TargetTracker()
    steering = SteeringController()
    forward = ForwardController()

    control_state = SharedControlState()
    command = SharedCommand()

    worker = DreameWorker(
        command
    )

    xbox = XboxThread(
        control_state,
        command,
    )

    worker.start()
    xbox.start()

    xbox.ready.wait(
        timeout=3
    )

    if xbox.error:
        raise xbox.error

    print("Y: autopilot ON/OFF")
    print("A: STOP")
    print("B: exit")
    print("Left stick: manual override")
    print("R keyboard: reset target")
    print(
        f"Auto forward limited to "
        f"{MAX_AUTO_FORWARD}"
    )

    last_time = (
        time.perf_counter()
    )

    fps = 0.0

    telemetry_exists = os.path.exists(TELEMETRY_PATH)
    telemetry_file = open(
        TELEMETRY_PATH,
        "a",
        newline="",
        buffering=1,
        encoding="utf-8",
    )
    telemetry = csv.writer(telemetry_file)

    if not telemetry_exists:
        telemetry.writerow([
            "timestamp",
            "state",
            "track_id",
            "confidence",
            "offset_x",
            "area_ratio",
            "velocity",
            "rotation",
            "source",
            "rtt_ms",
            "inference_ms",
            "fps",
        ])

    try:
        while True:
            if worker.error:
                raise RuntimeError(
                    f"Robot worker stopped: {worker.error}"
                ) from worker.error

            if xbox.error:
                raise RuntimeError(
                    f"Xbox worker stopped: {xbox.error}"
                ) from xbox.error

            (
                autopilot,
                exit_requested,
                manual_override,
                manual_velocity,
                manual_rotation,
            ) = control_state.snapshot()

            if exit_requested:
                break

            ok, frame = cap.read()

            if not ok:
                command.set(
                    0,
                    0,
                    "CAMERA_ERROR",
                )
                break

            frame_h, frame_w = (
                frame.shape[:2]
            )

            infer_started = (
                time.perf_counter()
            )

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[
                    CAT_CLASS_ID
                ],
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            infer_ms = (
                time.perf_counter()
                - infer_started
            ) * 1000.0

            detections = (
                extract_detections(
                    results[0]
                )
                if results
                else []
            )

            target = tracker.update(
                detections,
                frame_w,
                frame_h,
            )

            (
                autopilot,
                exit_requested,
                manual_override,
                manual_velocity,
                manual_rotation,
            ) = control_state.snapshot()

            if exit_requested:
                break

            desired_rotation = 0
            desired_velocity = 0

            if manual_override:
                steering.reset()
                forward.reset()

            elif (
                autopilot
                and target is not None
                and target["state"]
                == "TRACKING"
            ):
                desired_rotation = (
                    steering.update(
                        target["offset_x"]
                    )
                )

                desired_velocity = (
                    forward.update(
                        target,
                        desired_rotation,
                    )
                )

                command.set(
                    desired_velocity,
                    desired_rotation,
                    "AUTOPILOT",
                )

            elif autopilot:
                steering.reset()
                forward.reset()

                command.set(
                    0,
                    0,
                    "CV_STOP",
                )

            else:
                steering.reset()
                forward.reset()

                command.set(
                    0,
                    0,
                    "IDLE",
                )

            # ---------- visualization ----------
            for det in detections:
                x1, y1, x2, y2 = (
                    det["box"]
                )

                is_target = (
                    tracker.locked_id
                    == det["track_id"]
                )

                color = (
                    (0, 255, 0)
                    if is_target
                    else (120, 120, 120)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3 if is_target else 1,
                )

            cx = (
                frame_w // 2
            )

            for threshold, color in (
                (
                    CENTER_INNER,
                    (0, 180, 0),
                ),
                (
                    CENTER_OUTER,
                    (100, 100, 100),
                ),
            ):
                left = int(
                    frame_w / 2
                    - threshold
                    * frame_w / 2
                )

                right = int(
                    frame_w / 2
                    + threshold
                    * frame_w / 2
                )

                cv2.line(
                    frame,
                    (left, 0),
                    (left, frame_h),
                    color,
                    1,
                )

                cv2.line(
                    frame,
                    (right, 0),
                    (right, frame_h),
                    color,
                    1,
                )

            cv2.line(
                frame,
                (cx, 0),
                (cx, frame_h),
                (255, 255, 255),
                1,
            )

            if (
                target
                and target["center"]
            ):
                tcx, tcy = (
                    target["center"]
                )

                cv2.circle(
                    frame,
                    (tcx, tcy),
                    6,
                    (0, 255, 255),
                    -1,
                )

            now = (
                time.perf_counter()
            )

            dt = (
                now - last_time
            )

            last_time = now

            if dt > 0:
                inst = 1.0 / dt

                fps = (
                    inst
                    if fps == 0
                    else (
                        fps * 0.9
                        + inst * 0.1
                    )
                )

            actual_velocity, actual_rotation, source = (
                command.get()
            )

            telemetry.writerow([
                f"{time.time():.3f}",
                tracker.state,
                target["track_id"] if target else "",
                f"{target['confidence']:.4f}" if target else "",
                f"{target['offset_x']:.4f}"
                if target and target["offset_x"] is not None else "",
                f"{target['area_ratio']:.5f}" if target else "",
                actual_velocity,
                actual_rotation,
                source,
                f"{worker.last_rtt_ms:.1f}",
                f"{infer_ms:.1f}",
                f"{fps:.1f}",
            ])

            offset_text = "?"
            area_text = "?"

            if target:
                if (
                    target["offset_x"]
                    is not None
                ):
                    offset_text = (
                        f"{target['offset_x']:+.3f}"
                    )

                area_text = (
                    f"{target['area_ratio'] * 100:.1f}%"
                )

            lines = [
                (
                    f"AUTOPILOT: "
                    f"{'ON' if autopilot else 'OFF'}"
                ),
                f"CV: {tracker.state}",
                f"offset_x: {offset_text}",
                f"area: {area_text}",
                (
                    f"cmd V={desired_velocity:+d} "
                    f"R={desired_rotation:+d}"
                ),
                f"CONTROL: {source}",
                (
                    f"Dreame RTT: "
                    f"{worker.last_rtt_ms:.0f} ms"
                ),
                (
                    f"inference: "
                    f"{infer_ms:.0f} ms"
                ),
                (
                    f"loop FPS: "
                    f"{fps:.1f}"
                ),
            ]

            y = 26

            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                y += 23

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

            if key == ord("r"):
                tracker.reset()
                steering.reset()
                forward.reset()

                command.set(
                    0,
                    0,
                    "TARGET_RESET",
                )

    except KeyboardInterrupt:
        pass

    finally:
        print("\nStopping...")

        control_state.force_stop()

        command.set(
            0,
            0,
            "SHUTDOWN",
        )

        time.sleep(0.2)

        xbox.shutdown()
        worker.shutdown()

        cap.release()
        telemetry_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        sys.exit(1)
