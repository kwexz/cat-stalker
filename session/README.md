# Autonomous data-collection sessions

Goal: the human puts the P30 on the robot and walks away. The phone
streams, the PC persists, the robot keeps its Wi-Fi awake and talks
back through its speaker. No taps, no logcat watching, no cable.

## Roles

- **P30 (thin buffer):** runs detection as usual, keeps the screen on
  (`FLAG_KEEP_SCREEN_ON`), holds at most ~2000 telemetry rows in RAM
  and ~100 MB of event JPEGs on disk. Everything is deleted after the
  PC acks it. Event frames: detection transitions, every 5 s while
  DETECTED, low-confidence frames (max ~1/3 s).
- **PC (thick storage):** `session/auto_session.py` serves HTTP,
  writes `sessions/<stamp>/phone_telemetry.csv`,
  `robot_telemetry.csv`, `frames/`, `events.jsonl`, runs the Dreame
  keepalive, the watchdogs, and the speaker signals.

## Zero-touch flow

1. PC: `python -m session.auto_session --mode observe` (or `patrol`).
   It prints the `phone_url`, e.g. `http://192.168.1.239:18768`.
2. P30: just launch the app. It boots into NCNN Vulkan, broadcasts
   UDP discovery, finds the collector, and starts uploading by itself.
   The PC host field is only a manual fallback.
3. Human leaves. The robot signals when needed (see below).
4. Session ends by duration, Ctrl+C, or watchdog. Come back on sound.

Phone and PC must be on the same Wi-Fi. Windows Firewall must allow
inbound TCP on the collector port (default 8765/18768 in tests) and
UDP 8766 for discovery. The app uses plain HTTP on the LAN only
(`usesCleartextTraffic`, same trust level as ADB itself).

## Robot Wi-Fi sleep and reconnects

The Z10 Pro sleeps Wi-Fi while idle (ping signature: timeouts, then
`3789ms → 103ms → 5ms → 1ms`). `connect_with_wake_retries` therefore
tries 3 times, sending a ping burst between attempts to wake the
radio. If the robot moved to a new DHCP address, set `DREAME_IP`.

The phone does the same in reverse: after ~30 s of failed uploads the
uploader re-runs UDP discovery and migrates to the new collector
session by itself. No taps, no USB round-trip.

Tip for fewer USB trips: while on USB, run `adb tcpip 5555`, note the
phone IP (`adb shell ip route`), then `adb connect <ip>:5555` works
over Wi-Fi until the next reboot.

## Robot link and sounds

`DREAME_TOKEN` set → full link: `move(0,0)` keepalive at 1.5 Hz
(the Z10 Pro sleeps Wi-Fi otherwise), Quiet fan, speaker signals:

| Signal | Sound | Meaning |
|---|---|---|
| attention | `locate` (siid 7/aiid 1) | come to the robot/phone |
| done | `play_sound` ×1 (siid 7/aiid 2) | session finished, collect devices |
| error | `play_sound` ×2 | error stop, human needed |

`DREAME_TOKEN` missing → observe-only: phone telemetry is still
collected, but no motion, keepalive, or sounds. The log says so
explicitly.

## Motion policy

- `observe` (default): zero motion commands, keepalive only.
- `patrol`: slow in-place rotation sweep (±8, alternating every 20 s)
  for diverse capture angles. No translational driving in this stage.
- `follow`: closed-loop visual servo on the phone stream
  (`session/follow.py`, math ported from `cat_stalker.py`):
  3-of-5 acquisition → TRACKING (P+D steer, area-gated
  approach, forward conf gate 0.32 for P30 scores) →
  brief loss coasts on the frozen target (TEMP_LOST ≤0.6 s) →
  LOST_WAIT holds (1.5 s minimum, longer if the target was seen
  <6 s ago or is adjacent) → far+stale loss sweeps in place
  (one direction per episode, 20 s budget) → giveup + attention
  sound + SEARCHING with 10 s listen cycles. Rotation caps at 22,
  cruise at 40, slew-limited; speeds scale with measured FPS.
  Area thresholds are UNCALIBRATED for the low robot camera —
  remeasure from the mount before trusting distance.
  Requires robot link; without token it refuses and stays in observe.
- No cat in view is not the end: follow sweeps in place on start,
  then repeats sweep (20 s) / listen (10 s) cycles until the session
  ends or the cat is acquired. Only the first giveup sounds attention;
  repeats are logged. The program ends only on duration, watchdogs,
  dashboard stop, or Ctrl+C.
- Any motion command goes stale after 0.5 s (WATCHDOG → stop).
- Ctrl+C always ends with `(0,0)` + `stop_clean`.

## Speed governor (answers "adapt speed to inference?")

Yes — it helps stability, not detection quality. The loop runs on
fresh frames only (`STRATEGY_KEEP_ONLY_LATEST` on the phone, no
command queue on either side), and patrol speed scales with the
measured phone inference FPS: `scale = clamp(fps/8, 0.4, 1.0)`.
At 4 FPS the robot turns at half speed; stale FPS (>5 s) holds the
last safe scale; phone silence >15 s holds position `(0,0)` and
sounds attention; >60 s stops the session. This prevents the
overshoot/oscillation seen in early steering work. It does not fix
recall — that is model/data.

## Interception (how to stop it)

- Phone Stop button → uploads cease → hold in ~15 s, full stop ~60 s.
- PC Ctrl+C → immediate `(0,0)` + `stop_clean`, done sound.
- Physical: lift the robot or press its button.
- PC polls the phone stop flag the other way too
  (`GET /api/v1/session`), so a PC-side stop halts uploads.

## Watchdogs (all automatic)

- Phone heartbeat >15 s → attention sound; >60 s → error sound + stop.
- Robot `failures >= 5` → error sound + stop. Persistent failures
  (>10 s) halt patrol to keepalive + attention.
- Session duration reached → done sound + graceful stop
  (the PC tells the phone to stop via `GET /api/v1/session`).
- PC disk <1 GB → error sound + stop.

Honesty note: without Dreame odometry/pose we cannot observe blocked
wheels. Link-degraded handling above is the current proxy; true
motion-stuck detection needs the later map/pose stage.

## Mount placement (learned the hard way)

The phone mount must NEITHER cover the lidar turret NOR press on it:
pressure on the tower shifts something inside and the robot reports
the laser-sensor-cover prompt, then its blind obstacle logic can
reverse it into a wall — no PC command does that. Proven by contrast:
5-min handheld patrol → `device_fault=0` throughout, clean finish;
same commands with the pressing mount → fault + reverse + hang.
Mount low at the front, zero load on the turret, keep cliff sensors
clear.
Mount low at the front, keep cliff sensors clear. Patrol re-asserts
its rotation every 0.4 s (the motion slot expires non-zero commands
after 0.5 s; a single set would decay into keepalive zeros).

## Fault interlock

Every ~5 s the session polls `device_fault`/`device_status`. Any
nonzero fault (blocked laser, stuck bumper, …) forces `(0,0)` hold in
all motion modes, sounds attention once, and logs to
`robot_status.csv`. Motion resumes automatically when the fault
clears. This does not replace correct mounting — it only stops us
from fighting the robot's own safety logic.

## How the human sees progress (away from the PC)

- **Speaker first:** `locate` = come here, 1× `play_sound` = done,
  2× = error. No monitoring needed.
- **Dashboard:** `http://<pc-ip>:<port>/` (e.g. `http://192.168.1.239:8765/`)
  auto-refreshes every 5 s: rows, frames, heartbeat age, FPS, stop
  flag — plus a Stop button. Open from any browser on the LAN.
- **`sessions/<stamp>/STATUS.md`:** same numbers as text, rewritten
  every 5 s. Full history in `events.jsonl`.

## Automatic analysis (when it starts, what it does)

Off by default. Enable with `--analyze-every-min 15`: every N minutes
the session runs `session/analyze.py` on the live session dir, in a
background thread, never touching models or devices.

Each run appends an `analysis` event (visible in events.jsonl,
STATUS.md, dashboard) and writes `ANALYSIS.md` next to the data:

1. Telemetry stats: rows, DETECTED share, mean confidence/FPS, backends.
2. Frame dedup by perceptual hash (near-duplicates dropped).
3. Pseudo-labels with the production model; max-confidence per frame.
4. Borderline mining: frames in `[0.25, 0.5)` + contact sheet —
   these are the only frames a human may need to review.
5. Decision line, e.g. `clean session`, `N borderline frames need
   human review before any retraining`, `no cat activity worth
   training on`.

Retraining itself is deliberately NOT automatic yet: a new fine-tune
starts only after a human confirms the borderline set, and any
candidate must pass the false-positive gates (keyboard/cushions/wall
picture) plus a P30 runtime check before replacing COCO. That is the
next loop to automate, not this one.

## Verifying without the robot

```powershell
$env:DREAME_TOKEN = $null
python -m session.auto_session --mode observe --duration-min 1 --port 8765
```

Then point the app at the printed `phone_url` (or let discovery find
it) — rows land in `sessions/<stamp>/phone_telemetry.csv`.
