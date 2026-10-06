# Cat Stalker — instructions for coding agents

## Goal

Build a pet computer-vision / robotics project using a Dreame Z10 Pro robot vacuum as the mobile base.

Current goal:
1. Detect and track the user's real cat from a camera mounted on the robot.
2. Keep the cat centered in frame.
3. Follow it slowly while maintaining a safe distance.
4. Preserve Xbox manual override at all times.
5. Later: autonomous reacquisition/search, mapping and cat telemetry.

This is an experimental pet project. Prefer incremental, testable changes over large rewrites.

## Current hardware

- Robot: Dreame Z10 Pro
- Dreame model id: `dreame.vacuum.p2028`
- Firmware: `4.1.8_1156`
- Robot LAN IP currently: `192.168.1.124`
- Camera: external USB webcam physically mounted on the robot
- Phone mount (robot runs): LANDSCAPE only, verified 2026-10-06.
  Portrait is unverified: the steering-axis mapping depends on it.
- Camera index on current Windows machine: `CAMERA_INDEX = 1`
- Xbox controller connected to Windows
- Development machine: Windows
- Project directory used so far: `C:\Source\cat-stalker`
- Python virtualenv: `.venv`

Do NOT hardcode or commit the Dreame token. Read it from `DREAME_TOKEN`.

## Current software stack

- Python
- OpenCV
- Ultralytics YOLO
- YOLO model: `yolo11n.pt`
- ByteTrack via Ultralytics
- pygame for Xbox input
- `python-miio` installed from current GitHub master, not the old PyPI prerelease

Install python-miio using:
`python -m pip install "git+https://github.com/rytilahti/python-miio.git"`

## Important Dreame findings

Local miIO control works.

`DreameVacuum(...).info()` returns the robot correctly.

Relevant action mapping:
- `home`: siid 3 / aiid 1
- `locate`: siid 7 / aiid 1
- `start_clean`: siid 4 / aiid 1
- `stop_clean`: siid 4 / aiid 2
- `move`: siid 21 / aiid 1
- `play_sound`: siid 7 / aiid 2

The `move` action is used with:
- piid 1 = rotation
- piid 2 = velocity

The official Mi Home manual-control mode and our `move` action both activate the vacuum motor. It cannot currently be fully disabled in manual driving mode.
Use `robot.set_fan_speed(0)` to switch it to Quiet.

`get_property_by(4, 15)` returned `code: -1`; do not assume newer Dreame remote-control property layouts apply to Z10 Pro.

## Critical Wi-Fi behavior

The Z10 Pro aggressively sleeps its Wi-Fi while idle.

Observed idle ping behavior was extreme: replies were buffered for multiple seconds, e.g. 7s, 6s, 5s... and then a burst of 1–2 ms replies.

Signal itself was good:
- signal around -63 dBm
- avg ack around -62 dBm
- expected throughput around 51 Mbps

When `move(...)` commands are sent continuously, Wi-Fi wakes up and becomes stable:
- approximately 1–2 ms ping
- no packet loss in the test
- application RTT later observed around 18 ms

Read-only `get_property_by(...)` heartbeat did NOT keep it awake.

Therefore the control worker must send periodic `move(0, 0)` while idle.

OpenWrt 2.4 GHz was changed to:
- 20 MHz width
- fixed channel 6

Do not remove keepalive unless testing an alternative.

## Camera / CV findings

Initial detector worked surprisingly well with COCO `cat` class.

Known false positive:
- an image of a cat printed on a pillow

ByteTrack + stable target locking now rejects that pillow most of the time once the real cat is locked.

Known weaknesses:
- partially occluded cat can disappear
- strong backlight / bright source beside cat can distort the bounding box
- generic COCO detector is sensitive to lighting

Do not fine-tune yet unless there is evidence it is necessary.

Current camera settings that restored performance through USB extension:
- camera index 1
- requested MJPEG
- 640x480
- requested 30 FPS
- YOLO `imgsz=480`

Measured in a real run:
- YOLO inference ~33 ms
- whole loop ~12.4 FPS
- Dreame RTT ~18 ms

2026-10-06 phone bug: detector boxes are sensor-buffer coords
(rotation=90); overlay compensated, but SessionRecorder.offsetX used
buffer-X = world VERTICAL, so steering chased the cat's height with a
constant error and spun forever. Fixed with rotation-aware azimuth
offset (mirrors the overlay mapping, + = cat right of center).

This is already sufficient for the slow robot. Do not optimize FPS without a concrete need.

## Tracking logic

Use ByteTrack with persistent IDs.

Current conceptual states:
- SEARCHING
- TRACKING
- TEMP_LOST
- LOST_WAIT

Target acquisition requires multiple stable frames.
Brief target loss should stop the robot but preserve target lock long enough for reacquisition.
2026-10-06: full stop on every flickered frame caused drive-stop-sweep
stutter, so TRACKING degrades to TEMP_LOST coast (frozen target,
<=0.6 s, fresh frames only) instead of an instant stop. LOST_WAIT and
stale frames still stop.

Never keep driving forward on uncertain CV state.

## Control architecture

The application is intentionally split into independent loops/threads:

1. Xbox input loop:
   - ~100 Hz
   - must remain responsive even while YOLO inference blocks
   - moving the stick disables autopilot immediately

2. CV loop:
   - webcam -> YOLO -> ByteTrack -> smoothed target state

3. Robot worker:
   - keeps one long-lived `DreameVacuum`
   - accepts only the latest desired command
   - never queues stale commands
   - wakes immediately on command changes
   - sends periodic keepalive

Stale movement commands MUST NOT accumulate.

## Xbox controls

Current intended mapping:
- Y: autopilot on/off
- A: STOP and autopilot off
- B: exit
- left stick X/Y: manual drive and immediate autopilot override

Manual control has priority over CV.

## Steering behavior

Early controller oscillated left/right because of CV smoothing + command latency.

It was improved with:
- EMA smoothing
- inner/outer hysteresis around image center
- proportional + small derivative steering
- braking instead of immediate direction reversal after crossing center

V1.5 behavior:
- accurately centered the cat
- did not significantly overshoot
- reacquired after temporary occlusion
- but started turns too late and rotated too slowly

V1.6 increased steering responsiveness and added forward following.

## Current follow behavior

Latest tested V1.6:
- robot turns toward the cat
- approaches the cat successfully
- user reports it can probably approach closer

Current autonomous behavior should remain conservative:
- if target is too far off center: rotate first, do not drive
- 2026-10-06: that rule blocked ALL forward (cat sat at |offset|~0.58,
  gate was 0.13) — widened FORWARD_MAX_OFFSET to 0.65, speed already
  scales down to creep (v=4) at full turn
- search sweep is continuous one-direction per episode (exit side, else
  alternate); the old +/- flip netted ~zero ("stuck angle")
- 2026-10-06: alternating episodes retraced the same ~180 deg sector
  (r=10 x 20 s) while the cat sat behind in the unscanned half. Now:
  no-exit-side episodes CONTINUE the previous direction, coverage
  accumulates to 360+ across episodes; exit side still wins on fresh loss.
- 2026-10-06: faster + smoother motion. SEARCH_ROTATION 10->14,
  MAX_AUTO_ROTATION 14->18. Slew limits (5 rot / 4 vel per 8 Hz cycle)
  ramp AUTOPILOT/SEARCH instead of stepping on every flicker; full
  swing ~0.4 s. Stops, holds and giveups stay direct.
- 2026-10-06: cruise MAX_AUTO_FORWARD 8->10 (prototype parity).
  LOST_WAIT holds 1.5 s (CLOSE_HOLD) before sweeping so detector
  flicker reads as a pause, not a spin; far+real loss still sweeps.
- 2026-10-06: rotation deadzone. 75 s of r=6-7 held offset frozen
  (MIN 6 never centered; only |r|>=14 visibly turns). MIN_AUTO_ROTATION
  6->10. Sign verified correct (r=-14 drove off +0.6 leftwards, toward
  the cat). Watch for center overshoot oscillation next.
- 2026-10-06: "turns away" from a close cat was a lost race, not wrong
  direction: area 0.33->0.57 in 1 s while v=0 (THE CAT moved), then a
  walking cat held +0.84 edge against max turn (~9 deg/s at r=14 vs
  ~28 deg/s tangential at 2 m). MAX_AUTO_ROTATION 18->22 (prototype
  value) to shorten the lag. A walking cat still outruns rotation;
  verify centering on a sitting cat.
- 2026-10-06: close-range sweep orbit. A sitting cat at ~1.5 m flickers
  with multi-second gaps; every gap swept the robot into a side orbit.
  LOST_WAIT now sweeps only if unseen for SWEEP_GRACE_S=6 s (recency,
  not area); fresher losses hold. Verified by tools/test_follow_hold.py.
- 2026-10-06: cruise MAX_AUTO_FORWARD 10->20->40 (user: 2x twice).
  SLEW_VELOCITY 4->8 to keep up. STOP_AREA unchanged: watch for
  overshoot into the cat.
- if target is lost/uncertain: stop
- 2026-10-06 pillow run: v5 FIRES on the cat print at close range /
  low angle (54% det) despite passing pillow gates at test distance.
  sessions/20261006_202103 is marked NOT_FOR_TRAINING (hard negatives
  only, never cat positives) — check marker files before any retrain.
- use bbox area only as a rough apparent-distance proxy
- do not reverse autonomously yet

The next likely tuning task is reducing the stop distance / adjusting bbox area thresholds after measuring actual `area` values at useful distances.

## Safety / test discipline

During USB-webcam testing:
- cable is physically attached to a PC, so avoid cable winding
- keep speeds low
- Xbox STOP/manual override must remain available
- do not change multiple unrelated parameters at once
- print telemetry needed to diagnose behavior

Prefer exposing controller parameters near the top of the script for rapid experimentation.

## Near-term roadmap

1. Tune approach distance and forward-speed profile.
2. Improve steering/forward blending while preserving stability.
3. Record telemetry to CSV/JSON for tuning:
   timestamp, track_id, confidence, offset_x, bbox area, requested V/R, RTT, FPS.
4. Replace tethered webcam with an on-robot wireless camera source, likely an old Android phone first.
5. Consider segmentation only if bbox quality becomes a real blocker.
6. Later use robot map/pose for cat location telemetry.
7. Later implement LOST -> autonomous search/reacquire.
