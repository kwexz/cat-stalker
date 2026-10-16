# Measured results and experiment status

This page consolidates measurements already recorded in the repository. It does not turn a supervised prototype into a product claim. Dates and tuning history remain in [`AGENTS.md`](../AGENTS.md); detector lineage remains in [`data/p30_cat_training_results.md`](../data/p30_cat_training_results.md).

## Test platforms

| Platform | Role | Relevant setup |
|---|---|---|
| Huawei P30 | Current edge camera/inference node | Android 10, rear CameraX stream, landscape robot mount, NCNN Vulkan, 384 px FP16 model |
| Windows PC | Collector and controller | Receives telemetry/event JPEGs; control loop at 8 Hz; sends only the latest desired command |
| Dreame Z10 Pro | Mobile base | Model `dreame.vacuum.p2028`, local miIO action calls; private address and token supplied through environment variables |
| USB webcam + Windows PC | Earlier prototype | MJPEG 640×480 requested at 30 FPS, YOLO `imgsz=480`, ByteTrack and Xbox override |

## Runtime measurements

| Measurement | Result | Evidence / interpretation |
|---|---:|---|
| P30 active NCNN 384 FP16 inference | ~100–120 ms sustained | Device UI/runtime trials summarized in [`android/README.md`](../android/README.md); the v5 report also records roughly the same 100–120 ms range. |
| P30 active NCNN analysis/inference rate | ~8–9 FPS sustained | Recommended operating point. A transient touch/governor boost of 12–15 FPS is explicitly excluded from the baseline. |
| P30 TFLite CPU | ~150 ms initially, ~250 ms after ~30 s; ~5.1 → ~3.3 FPS | Rejected for sustained use because of thermal throttling. |
| P30 NCNN 480 FP16 | ~150–180 ms; ~5.8–6.2 FPS | Rejected as default: insufficient distant-cat recall gain for the throughput loss. |
| Earlier Windows YOLO inference | ~33 ms | USB-camera prototype, not the current phone path. |
| Earlier Windows whole CV loop | ~12.4 FPS | Adequate for the slow platform; no need to treat it as a phone benchmark. |
| Dreame LAN ping while awake | ~1–2 ms | Continuous `move` traffic wakes the Z10 Pro Wi-Fi; read-only heartbeat did not. |
| Dreame application RTT while awake | ~18 ms | Observed command round trip, not an upper-bound guarantee. |
| Controller update rate | 8 Hz | `session/auto_session.py`. |
| Idle keepalive rate | 1.5 Hz | `move(0,0)`; this is still a physical device command. `--no-robot` disables the link entirely. |
| Non-zero command expiry | 0.5 s | Stale command becomes `(0,0, WATCHDOG)`. |
| Fresh-frame limit for forward motion | 0.5 s | Forward motion is blocked when telemetry is older. |

Phone rates include the real CameraX analysis path; inference milliseconds and FPS are not necessarily mathematical reciprocals because conversion, scheduling, frame dropping, and thermal/governor behavior contribute separately.

## Detector evaluation

The active custom model is the v5 Huawei P30 cat detector exported to NCNN 384 FP16.

| Model / experiment | Precision | Recall | mAP50 | mAP50–95 | Runtime decision |
|---|---:|---:|---:|---:|---|
| COCO `yolo11n.pt` baseline on initial local split | 0.202 | 0.347 | 0.081 | 0.077 | Historical baseline |
| Fine-tune v1 | 0.959 | 0.959 | 0.972 | 0.755 | Rejected on device: severe pillow/keyboard false positives |
| Fine-tune v2 | 0.920 | 0.934 | 0.952 | 0.703 | Rejected during runtime/preprocessing investigation |
| Fine-tune v3 | 0.932 | 0.918 | 0.951 | 0.695 | Candidate only |
| Fine-tune v4 | 0.951 | 0.898 | 0.947 | 0.702 | Candidate only |
| **Fine-tune v5** | **0.937** | **0.939** | **0.958** | **0.773** | Active on-device model after runtime tests |

The split is disjoint by source video, which is stronger than a random frame split, but all source material is still a small local collection from the same apartment context. The training report records 344 v5 training frames (228 cat-positive and 116 negative); it does not record a final v5 validation frame count, so no count is inferred here.

> **Generalization warning:** these precision/recall/mAP values describe only the local held-out set. They do not guarantee similar accuracy for another cat, home, camera, lighting condition, viewpoint, printed-cat object, or deployment. The v1 failure demonstrates why offline metrics alone are insufficient: it scored very highly and still produced unacceptable real-device false positives.

No public benchmark, confidence interval, cross-home test, or statistically representative safety evaluation has been performed.

## Current control configuration

Values below document the current implementation; they are not recommendations for other robots and must not be interpreted as SI units.

| Area | Parameter | Value |
|---|---|---:|
| Acquisition | lock window / required hits | 5 / 3 |
| Loss handling | `TEMP_LOST_S` / `LOST_S` | 0.6 s / 1.5 s |
| Smoothing | position / area EMA alpha | 0.60 / 0.25 |
| Steering | inner / outer center thresholds | 0.05 / 0.10 normalized offset |
| Steering | P / D gain | 24.0 / 3.0 |
| Steering | min / max autonomous rotation command | 10 / 22 |
| Forward | start / stop bbox area ratio | 0.095 / 0.135 |
| Forward | maximum command | 40 |
| Search | rotation command / episode budget | 14 / 20 s |
| Slew | rotation / velocity per 8 Hz update | 5 / 8 |

The controller uses bbox area only as an apparent-distance proxy. The values are command units and normalized image ratios, not metres, metres per second, degrees per second, or calibrated distance.

## Behavior evidence

### Demonstrated in supervised runs

- End-to-end phone detection → telemetry → PC control decision → local robot command path.
- Centering and approach toward a calm sitting cat.
- Temporary-loss handling and reacquisition in some occlusion cases.
- Full stop/hold for stale frames, uncertain target state, fault interlock, and watchdog conditions.
- Single-slot command delivery without stale movement backlog.

The repository's real setup photo, [`docs/cat_robot_phone.jpg`](cat_robot_phone.jpg), shows the mounted P30 and a live `DETECTED` overlay. It proves the physical setup and phone detector display, but a still image alone does not prove robot movement or closed-loop behavior.

### Explicit moving-cat limitation

A close walking cat was observed at about 28°/s apparent tangential motion around 2 m, while a representative `rotation=14` command turned the platform at roughly 9°/s. The detector/controller/platform chain therefore cannot reliably keep a moving cat centered. The target may reach the frame edge, be lost, and trigger hold/search behavior. A seated-cat centering run is the valid current demonstration; “autonomously follows a moving cat” is not.

### Remaining unknowns

- Repeatability across homes, cats, mounting geometry, and lighting.
- Metric stopping distance and overshoot distribution.
- End-to-end latency distribution from exposure to physical yaw; only component observations are recorded.
- Long-duration thermal and battery behavior beyond the documented trials.
- True wheel-stuck state, because robot pose/odometry is not integrated.
- Safety around arbitrary obstacles, pets, stairs, and people.

## Reproducing what can be reproduced publicly

1. Run the hardware-free checks: `python -m unittest discover -s tests -v`.
2. Start the collector with `python -m session.auto_session --mode observe --no-robot --duration-min 5`.
3. Inspect the dashboard and generated session schema without connecting any device.
4. To reproduce detector metrics, supply the private source media locally and follow [`data/DATASET_WORKFLOW.md`](../data/DATASET_WORKFLOW.md). The media are intentionally not public, so exact third-party reproduction of the reported model metrics is not currently possible.
5. Hardware behavior requires equivalent devices and direct human supervision;
   results will depend on the environment. The repository photo documents the
   physical setup, while this report records the measured runtime evidence.
