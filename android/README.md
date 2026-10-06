# Cat Stalker Android Vision

This is a self-contained Android prototype. It performs local cat detection only:

```text
rear camera -> CameraX latest frame -> YOLO11n runtime -> cat bbox + telemetry overlay
```

It does not control the Dreame, track targets, communicate with a PC, or send camera frames to a network service.

## Runtime choice

The app has two switchable local runtimes: TensorFlow Lite CPU at 480 and NCNN Vulkan at 384. TFLite is retained as a correctness baseline because Ultralytics directly exports YOLO11 models to it. NCNN uses the official Android Vulkan shared library and a minimal JNI decoder for a separately exported YOLO11n 384 graph. Each runtime receives only the input size it was exported for.

The P30's Android 10 / Mali stack could not load the current TFLite GPU delegate, so it is not offered. NCNN confirms Vulkan initialization on the P30; the UI reports `NCNN Vulkan` only when this backend is active.

## Model export

The app expects these untracked model assets:

```text
android/app/src/main/assets/yolo11n_float32.tflite
android/app/src/main/assets/yolo11n_ncnn_model/model.ncnn.param
android/app/src/main/assets/yolo11n_ncnn_model/model.ncnn.bin
```

NCNN's prebuilt Vulkan headers and `libncnn.so` are also intentionally untracked. Fetch them on Windows before a clean Android build:

```powershell
.\tools\fetch_ncnn_android.ps1
```

Current Ultralytics versions use the unified LiteRT exporter, which supports Linux x86_64 and macOS, not a native Windows host. From WSL2 Ubuntu (or Linux/macOS), with Ultralytics installed, run from the repository root:

```bash
python tools/export_android_model.py
```

The script exports the existing `yolo11n.pt` with `imgsz=480`, float32 RGB input, no embedded NMS, and NCNN. Both decoders expect the YOLO11 raw head `[1, 84, 4725]` (`xywh + 80 COCO scores`). LiteRT returns normalized `xywh`, while NCNN returns pixel `xywh`; both paths convert to source-frame coordinates before the shared overlay. They supply RGB normalized to `[0, 1]`, extract COCO class `15` (`cat`) at confidence `0.28`, and apply class-local NMS at IoU `0.45`.

Binary `.tflite` assets are ignored by git. Re-run the export after changing the source model.

## Build and install

Requirements: Android Studio / SDK Platform 35, JDK 17+, and a USB-connected Huawei P30 with USB debugging enabled.

```powershell
cd android
.\gradlew.bat assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Or open `android/` in Android Studio, sync Gradle, place the exported asset first, and run the `app` configuration on the P30.

On the phone, grant the Camera permission. Point the rear camera at a real cat, wait for the live preview, then use `Switch to GPU` and `Switch to CPU`. The switch recreates the local interpreter on the single CameraX analysis thread; frames are not queued.

## Pipeline and metrics

- CameraX `ImageAnalysis` requests `480x480` and uses `STRATEGY_KEEP_ONLY_LATEST`. The detector still letterboxes to 480x480, so this removes needless high-resolution YUV conversion work.
- A single analysis executor owns conversion, inference, and detector swapping. A busy inference drops stale frames instead of accumulating latency.
- The screen shows detection state, confidence, normalized horizontal offset, bbox area ratio, per-frame end-to-end detector time, smoothed analysis/inference FPS, and actual selected backend. Logcat separately reports YUV-to-RGB conversion time.
- The app does not write telemetry to files. It emits one compact `CatStalkerVision` diagnostic line per second to logcat with bbox coordinates, rotation, inference time, FPS, and backend. Android's system log buffer is ring-buffered independently of this app.

## P30 benchmark record

Actual measurements must be collected on the target Huawei P30; no device performance is fabricated in source control.

| Mode | Backend reported by app | Inference ms | Inference FPS | Analysis FPS | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| TFLite CPU release | CPU | ~150 ms initially; ~250 ms after ~30 s | ~5.1 initially; ~3.3 sustained | ~5.1 initially; ~3.3 sustained | Confirmed P30 thermal throttling. Do not use for sustained operation. |
| NCNN Vulkan 480 | NCNN Vulkan | ~120-140 ms | ~4.0 | ~4.0 | Confirmed on P30 with the visible cat. |
| NCNN Vulkan 320 + bitmap conversion | rejected | ~80-100 ms | ~4.7-5.0 | ~4.7-5.0 | Invalid: the fixed 480 NCNN graph was fed 320 data, producing false positives and misplaced boxes. Do not use this result. Export a real 320 model before benchmarking this size. |
| NCNN Vulkan 384 + bitmap conversion | NCNN Vulkan 384 bitmap | ~90-100 ms | ~4.7-4.9 | ~4.7-4.9 | Confirmed on P30: bbox is sufficiently aligned and no false positive occurred without a cat. Overall FPS is limited by Kotlin YUV-to-RGB conversion. |
| NCNN Vulkan 384 + native YUV | NCNN Vulkan 384 native YUV | ~100 ms | ~7.5-8.2 | ~7.5-8.2 | Confirmed on P30: cat detection and bbox work. Active NCNN default. |
| NCNN Vulkan 384 + native YUV release | NCNN Vulkan 384 native YUV | ~100-130 ms | ~8-9 | ~8-9 | Confirmed P30 sustained release benchmark. Recommended current operating point. |
| NCNN Vulkan 384 FP16 + native YUV | NCNN Vulkan 384 FP16 native YUV | ~100-120 ms sustained; ~60-70 ms after touch | ~8-9 sustained; ~12-15 after touch | ~8-9 sustained; ~12-15 after touch | FP16 model is valid, but does not improve distant-cat recall. Touch boost is a transient device governor effect, not a usable control baseline. |
| NCNN Vulkan 480 FP16 + native YUV | ~150-180 ms | ~5.8-6.2 | ~5.8-6.2 | P30 daylight test: bbox/no-cat behavior remained correct. Sitting cat is detected at a somewhat greater distance, but neither posture is reliable at 4 m+. Lying cat remains unreliable. Not enough recall improvement to justify the FPS loss as an always-on mode. |
| NCNN Vulkan 320 + native YUV | NCNN Vulkan 320 native YUV | ~80-110 ms | ~10-11 | ~10-11 | Confirmed on P30: bbox works and no-cat behavior is correct, but a cat around 3.5 m away is sometimes missed even in good lighting. This is too much recall loss for the primary mode. |
| NCNN Vulkan 352 + native YUV | NCNN Vulkan 352 native YUV | ~100-120 ms | ~8.5-9.5 | ~8.5-9.5 | Correct bbox/no-cat behavior, but still misses the cat around 3.5 m. Recall loss is too high for the primary mode. |
| Native YUV prototype | experimental | ~110-130 ms | ~7.5-8.5 | ~7.5-8.5 | Rejected: direct-plane conversion corrupted detector input (`max_conf=0.000`), so cats were not detected. Removed from active code. |

The P30 result proves that Vulkan is available. In the release build, NCNN Vulkan 384 plus native YUV sustains about 8-9 FPS at roughly 100-130 ms and is the only candidate for sustained operation. TFLite CPU initially reaches around 150 ms / 5.1 FPS, then throttles after about 30 seconds to 250 ms / 3.3 FPS; it remains a correctness baseline only. The dedicated 320 and 352 exports are faster but miss the cat at about 3.5 m. Touching the P30 display temporarily raises performance to about 60-70 ms / 12-15 FPS, consistent with Huawei's short interaction CPU/GPU boost; an app must not rely on this for control. Android sustained-performance mode was requested and produced no measurable improvement on this P30/EMUI build, so it is retained as harmless standard behavior but not considered an optimization.

The 480 daylight test improves a sitting cat's maximum detection distance somewhat, but not enough to make cats at 4 m+ reliable and not enough to compensate for its 5.8-6.2 FPS. Lighting differs from earlier indoor-light benchmarks, so recall comparisons need controlled scenes before treating small changes as model effects. The quality-first operating point remains 384 native YUV for now; the next quality improvement should be a small P30-camera dataset that includes sitting and lying poses at measured distances under indoor and daylight illumination.

## P30 Fine-Tune Candidate

`p30_cat_384_fp16_ncnn_model` was a candidate runtime model trained from the reviewed P30 frames. It has a one-class output head (`cat` is class `0`), unlike COCO's 80-class output where cat is class `15`. Although its video-held-out metrics were high and P30 performance was about 100 ms / 8.5-9.5 FPS, runtime validation rejected it: gray pillows, gray keyboards, and similar objects were repeatedly detected as cats, while 4 m+ recall did not improve. The model overfit because the reviewed set contained only one negative frame. It is not an active runtime model. Return to the COCO 384 model until a new dataset includes deliberate hard negatives and a validation split that contains them.

`p30_cat_v2_384_fp16_ncnn_model` adds 60 confirmed hard-negative frames and holds out 25 negative validation frames. It is installed for the next P30 runtime test. Validate the original gray pillow and gray keyboard failures plus no-cat scenes before judging its distant-cat behavior.

`p30_cat_v3_384_fp16_ncnn_model` uses a balanced 165-cat/80-negative train split and is the current P30 candidate. It must be tested with bilinear native YUV against the keyboard, cushions, wall picture, lying cat, and no-cat scenes. Its v3 validation score is reported in `data/p30_cat_training_results.md`.

P30 runtime rejected v3: keyboard confidence increased to `0.6-0.8`, more cushions became false positives at `0.5+`, and cat confidence still ranged from `0.4-0.9`. The wall picture no longer triggered, but this does not justify the broader regression. The active runtime returns to COCO 384. Future custom training must use manually selected and reviewed hard-negative images of the exact keyboard, cushions, and wall picture rather than more uniformly sampled negative video frames.

`p30_cat_v4_384_fp16_ncnn_model` adds nine exact false-positive P30 captures (wall artwork, keyboard/work desk, cushions, cat tower, slippers), repeated as targeted empty-label training examples. It is the current candidate and must be tested against those exact scenes plus the lying cat before it can replace COCO.

## Targeted Hard-Negative Capture

The active COCO app includes `Save Hard Negative`. When a real false positive is visible, keep the object in frame, tap this button once, and the exact preview frame is saved on the P30 under `Pictures/CatStalker/HardNegatives`. This is intentionally manual: save only the keyboard, specific cushions, and wall-picture patch that actually trigger. The files can be collected later with `adb pull /sdcard/Pictures/CatStalker/HardNegatives data/hard_negative_captures`.

## Autonomous sessions (thin phone buffer)

The app is a thin buffer during training/tuning: at most ~2000
telemetry rows in RAM and ~100 MB of event JPEGs in cache, deleted
after the PC acks them. It keeps the screen on while visible, reports
battery/disk in the heartbeat, and polls the PC stop flag.

Zero-touch steady state, verified on the P30:

1. PC runs `python -m session.auto_session` (see `session/README.md`).
2. Launch the app — nothing else. It boots into NCNN Vulkan,
   finds the collector via UDP discovery, and starts uploading.
3. The `PC host` field + Start/Stop button are manual fallback only.

Details: `session/README.md`, wire protocol: `session/protocol.py`.

## Reference Device Result

An Honor BVL-N49 (Android 16, Snapdragon `pineapple` / Adreno) was used only as a one-time performance reference, not as an MVP target. With the same release APK and NCNN Vulkan 384 native-YUV path it reached about `10-20 ms` and `30 FPS`; cat detection was also slightly better than the P30. This confirms the Android pipeline is not fundamentally limited to P30 performance. The MVP target remains the Huawei P30 and its sustained 384 native-YUV result.

## Backlight metering

Full-frame auto-exposure meters the bright window, so a backlit cat
exposes as a black silhouette (seen live: cat in a doorway against
daylight, detection collapsed from 15.8% to 0.8% in that spot). While
TRACKING, the app meters AE on the bbox center via
`FocusMeteringAction.FLAG_AE` (throttled, auto-cancel; reset on loss).
Best-effort per device. The backlit event frames from that session are
kept for the dataset regardless.

## Known limits

- This is detection-only: there is no ByteTrack, stable target lock, motion control, or robot integration.
- The overlay uses the rear-camera `PreviewView` center-crop transform. YOLO boxes naturally have a small semantic margin around a cat; tune neither detector thresholds nor box geometry until several distances and lighting conditions have been checked.
- The current Kotlin YUV-to-RGB conversion is the largest remaining CPU-side cost. The next optimization should replace it with a native CameraX/YUV path only after the controlled NCNN benchmark is recorded.
