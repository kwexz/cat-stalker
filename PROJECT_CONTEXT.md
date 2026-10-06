# Cat Stalker — project handoff

## 1. Original idea

The project started as a computer-vision pet project. The final direction became an autonomous robot that follows a cat around an apartment and eventually reports spatial/activity telemetry.

Instead of building a mobile robot from Arduino parts, an existing Dreame Z10 Pro robot vacuum is being reused as a mobile robotics platform.

Why this is attractive:
- motors and wheel control already exist
- battery and docking already exist
- lidar / navigation hardware already exists
- obstacle sensors already exist
- Wi-Fi exists
- only perception and higher-level behavior need to be added

For the current prototype, CV runs on a Windows PC and the camera is a USB webcam mounted to the Dreame.

## 2. Dreame Z10 Pro integration

Robot:
- Dreame Z10 Pro
- model: `dreame.vacuum.p2028`
- firmware: `4.1.8_1156`
- current LAN IP: `192.168.1.124`

A local miIO token was obtained separately. Do not store it in source control.

PowerShell environment:
```powershell
$env:DREAME_TOKEN="..."
```

Local device communication test succeeded:
```python
from miio.integrations.dreame.vacuum.dreamevacuum_miot import DreameVacuum

d = DreameVacuum(
    "192.168.1.124",
    os.environ["DREAME_TOKEN"],
    model="dreame.vacuum.p2028",
)

print(d.info())
```

It returned:
`dreame.vacuum.p2028 v4.1.8_1156 ... @ 192.168.1.124`

The installed `python-miio` PyPI prerelease had CLI bugs. The working setup uses current GitHub master:
```powershell
python -m pip uninstall python-miio -y
python -m pip install "git+https://github.com/rytilahti/python-miio.git"
```

### Action mapping discovered

```python
{
    'home': {'siid': 3, 'aiid': 1},
    'locate': {'siid': 7, 'aiid': 1},
    'start_clean': {'siid': 4, 'aiid': 1},
    'stop_clean': {'siid': 4, 'aiid': 2},
    'reset_mainbrush_life': {'siid': 9, 'aiid': 1},
    'reset_filter_life': {'siid': 11, 'aiid': 1},
    'reset_sidebrush_life': {'siid': 10, 'aiid': 1},
    'move': {'siid': 21, 'aiid': 1},
    'play_sound': {'siid': 7, 'aiid': 2}
}
```

Manual move params:
```python
[
    {"piid": 1, "value": str(rotation)},
    {"piid": 2, "value": str(velocity)},
]
```

### Manual cleaning / fan

Sending `move` wakes the robot and allows manual movement, but also activates suction.

The official Mi Home manual driving behaves the same way.

Quiet mode works:
```python
robot.set_fan_speed(0)
```

The fan does not fully turn off in current manual mode.

A test of:
```python
d.get_property_by(4, 15)
```
returned:
```text
[{'siid': 4, 'did': '4-15', 'piid': 15, 'code': -1}]
```

So do not assume that a newer Dreame remote-control property at 4/15 is usable on this model.

## 3. Network investigation

Initial local control was unreliable.

Windows -> router ping was good.

Router -> Dreame was terrible while robot was idle.

Example behavior:
```text
seq=0  ~7255 ms
seq=1  ~6255 ms
seq=2  ~5255 ms
...
seq=8  ~1.5 ms
```

A later burst showed ~12 seconds down to normal latency.

OpenWrt station data for Dreame:
```text
signal:       -63 dBm
signal avg:   -63 dBm
last ack:     -63 dBm
avg ack:      -62 dBm
tx bitrate:   58.5 Mbit/s
rx bitrate:   6.0 Mbit/s
expected throughput: ~51.6 Mbps
```

This led to the key finding:

### The robot is not suffering from weak RF; it aggressively sleeps Wi-Fi while idle.

Tests:
- reading a property once per second did NOT prevent sleep
- continuously sending `move(0,0)` DID prevent sleep
- while active/manual-driving:
  - ping became ~1–2 ms
  - packet loss disappeared
  - later application RTT was ~18 ms

So keepalive must use `move(0,0)`.

OpenWrt was also changed from 40 MHz to:
- 20 MHz
- channel 6 fixed

VPN was ruled out as the root cause.

## 4. Xbox teleoperation

Xbox controller was added through pygame.

An early synchronous implementation felt laggy because:
- CV and joystick were in the same loop
- robot calls waited for responses
- idle robot worker slept too long

Architecture was changed to:
- Xbox thread ~100 Hz
- latest-state command slot, no queue
- robot worker wakes immediately when command changes
- CV loop independent
- manual control always overrides autopilot

This made teleoperation responsive.

Intended controls:
```text
Y               autopilot toggle
A               stop + autopilot off
B               exit
left stick X/Y  manual steering/throttle, overrides autopilot
```

## 5. Computer vision evolution

### V0.1
`yolo11n.pt`, generic COCO class `cat`.

It detected the real cat well.

Main false positive:
- cat image printed on a pillow

Other weaknesses:
- sometimes misses partial/occluded cat
- sensitive to difficult lighting/backlight
- bbox can include nearby bright source

### V0.2
Added ByteTrack and stable target lock.

This solved most pillow switching:
- once real cat gets a stable track id, target lock stays on it
- brief detection misses do not immediately release target

Typical CV states:
```text
SEARCHING
TRACKING
TEMP_LOST
LOST_WAIT
```

### Webcam performance

After adding a USB extension, loop FPS initially fell near 5.

The code was changed to explicitly request:
- MJPEG
- 640x480
- 30 FPS camera
- YOLO `imgsz=480`

After that, measured UI telemetry was approximately:
```text
Dreame RTT: 18 ms
inference: 33 ms
loop FPS: 12.4
```

This is considered sufficient.

Current external webcam device:
```python
CAMERA_INDEX = 1
```

Remember this in future versions.

## 6. First closed-loop steering

Webcam was physically taped to the robot.

Initial autopilot only rotated; forward velocity was hard-disabled.

The robot could:
- detect cat
- lock target
- rotate toward it
- stop around image center
- reacquire after short occlusion

An early controller oscillated left/right.

The stabilizing changes were:
- EMA target smoothing
- inner/outer center hysteresis
- P + small D steering
- after crossing center, brake instead of instantly reversing

By V1.5:
- centering was fairly precise
- it did not noticeably overshoot center
- it could reacquire after occlusion
- but turning started too late and was too slow

## 7. Current V1.6

V1.6 made steering more aggressive and added slow forward following.

Relevant values:
```python
CAMERA_INDEX = 1

MAX_AUTO_ROTATION = 22
MAX_AUTO_FORWARD = 10

CONFIDENCE = 0.28
IMAGE_SIZE = 480

EMA_ALPHA_POSITION = 0.60
EMA_ALPHA_SIZE = 0.25

CENTER_INNER = 0.05
CENTER_OUTER = 0.10

KP_ROTATION = 24.0
KD_ROTATION = 3.0
MIN_AUTO_ROTATION = 6

FORWARD_START_AREA = 0.095
FORWARD_STOP_AREA = 0.135
FORWARD_MAX_OFFSET = 0.13
FORWARD_MIN_CONFIDENCE = 0.38
```

Forward controller behavior:
- if target is lost -> stop
- if confidence too low -> stop
- if cat too far off center -> stop forward motion and steer first
- if bbox area is small enough -> drive forward slowly
- if bbox area grows past stop threshold -> stop
- hysteresis avoids forward/stop chatter
- forward speed is reduced while steering
- no autonomous reverse yet

Latest real-world result from user:
> robot turned toward the cat and crept up to it successfully

User says it can probably approach closer.

So the immediate next tuning task is likely:
- observe `area` at desired real distance
- increase `FORWARD_STOP_AREA`
- possibly increase `FORWARD_START_AREA` correspondingly
- consider a smoother speed-vs-area controller instead of binary start/stop hysteresis

## 8. What not to prematurely optimize

Do not yet:
- train a custom cat model
- switch to segmentation solely because it exists
- chase 30 FPS
- rewrite low-level robot control
- implement SLAM from scratch

Existing quality is already enough to learn more from physical testing.

Segmentation becomes worthwhile if bbox area proves too unstable for distance control, especially under occlusion/backlight.

## 9. Next architecture steps

### Immediate
1. Make follow distance configurable at runtime.
2. Log telemetry:
   - timestamp
   - target track id
   - confidence
   - offset_x
   - bbox area
   - requested velocity
   - requested rotation
   - robot RTT
   - inference time
   - loop FPS
3. Tune approach behavior from logged data.

### Next hardware phase
Replace tethered USB webcam with a mobile camera:
preferred initial approach:
- old Android phone mounted on robot
- stream video over Wi-Fi to PC
- keep CV on PC at first

Later:
- move inference onto phone if useful

### Later robotics phase
Use Dreame map / pose if accessible to:
- map cat trajectory
- room-level telemetry
- heatmaps
- last-known-position search
- LOST -> search -> reacquire behavior

Potential eventual product behavior:
```text
detect cat
-> follow
-> cat disappears
-> retain last known position
-> autonomous search
-> reacquire
-> continue follow
```

## 10. Safety constraints for current tethered setup

Current webcam is physically connected to the PC by USB extension.

Do not allow free autonomous roaming yet.

Risks:
- winding USB cable around robot
- cable entering brushes/wheels
- robot dragging PC/peripherals

Keep test area small and maintain Xbox manual override.

## 11. Current project philosophy

Prefer a sequence of working increments:
- detect
- track
- rotate
- follow
- measure
- tune
- untether camera
- map
- search

Each stage should generate data before adding complexity.
