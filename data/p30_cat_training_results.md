# P30 Cat Fine-Tune v1

## Data

- Source: nine Huawei P30 rear-camera videos recorded in the apartment.
- Extracted frames: 216.
- Video-disjoint split: 168 training frames from seven videos; 48 validation frames from two unseen videos.
- Human review: 54 difficult/diverse frames. Decisions: 28 kept bootstrap boxes, 25 manually drawn/replaced boxes, 1 false box removed.

## Training

- Base model: `yolo11n.pt`.
- Input: 480.
- Training: CPU, batch 8, up to 60 epochs, early stopped after epoch 53.
- Augmentation: horizontal flip plus modest HSV, translation, and scale; no mosaic or mixup.

## Validation Comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| COCO baseline `yolo11n.pt` | 0.202 | 0.347 | 0.081 | 0.077 |
| P30 fine-tune v1 | 0.959 | 0.959 | 0.972 | 0.755 |

These validation videos are held out by source video, not by individual frame. They still come from the same apartment session, so P30 runtime testing remains mandatory before replacing the Android model.

## Result

Candidate weights: `runs/p30_cat/yolo11n_p30_v1/weights/best.pt`.

## P30 Runtime Rejection

The candidate was exported to NCNN 384 FP16 and tested on P30. It ran at about 100 ms and 8.5-9.5 FPS, but must not be used:

- It produced severe false positives on a gray pillow, gray keyboard, and similarly colored or sized objects.
- It did not improve detection beyond 4 m.

The validation split had only one reviewed negative frame and therefore did not represent these hard negatives. This is a textbook small-dataset overfit: high held-out-video metrics from one apartment session did not predict real room behavior. The Android runtime must return to the COCO model.

The next data iteration needs deliberately collected and reviewed hard-negative frames: gray pillows, keyboards, blankets, printed cats, clutter, and empty rooms. Do not retrain from the current labels alone.

## Fine-Tune v2: Hard Negatives

Five additional P30 videos supplied 60 confirmed cat-free hard-negative frames, including keyboards, gray pillows, blankets, cushions, furniture, floors, and empty rooms. The rebuilt split contains 204 train frames (165 cat, 39 negative) and 72 validation frames (47 cat, 25 negative). Hard-negative videos are held out by source video.

Validation for `runs/p30_cat/yolo11n_p30_v2_hard_negatives/weights/best.pt`:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.920 | 0.934 | 0.952 | 0.703 |

The lower score than v1 is expected because validation now includes true negatives. This candidate still needs P30 runtime validation against the original pillow and keyboard failure cases before it can replace COCO.

## P30 Runtime Rejection v2

The initial P30 native-YUV test rejected v2 because it failed to detect the real cat even at close range. An equivalence check then compared the `.pt`, NCNN FP32, and NCNN FP16 outputs on the same reviewed image: the export is correct (`0.800420` PyTorch confidence versus `0.800835` FP16 NCNN). The remaining suspect is Android's native YUV preprocessing rather than training or export. The next P30 test runs the same v2 NCNN model with the previously verified CameraX bitmap conversion. Do not collect more data until this test resolves the preprocessing mismatch.

Bitmap-preprocessed P30 v2 detects the cat and suppresses the gray pillow, proving the one-class model is not completely broken. A gray keyboard remains a false positive, which needs more keyboard negatives later. The native path used nearest-neighbor resize while CameraX bitmap uses bilinear scaling, so the next candidate changes native YUV sampling to bilinear before any dataset decision.

## P30 Runtime Result v2 With Bilinear Native YUV

Native bilinear preprocessing restores cat detection at about 110 ms / 8-9 FPS. It does not yet meet the false-positive requirement: the gray keyboard, some (but not all) cushions, and a color/size-similar patch in a wall picture are still detected as cats. This confirms the remaining problem is model calibration and hard-negative coverage, not export or Android preprocessing.

Before changing the training set again, measure the displayed confidence for each of the three false-positive objects and for a real cat at close and far distances. That establishes whether a higher confidence threshold is a safe immediate mitigation or would suppress the distant cat along with false positives.

## Confidence Decision

P30 measurement shows keyboard, wall-picture patch, cushion, and lying cat all overlap at `0.3-0.5`; sitting/walking cats score `0.7-0.9`. Raising the threshold would remove exactly the difficult lying-cat cases the custom model is meant to improve, so threshold-only mitigation is rejected. Fine-tune v3 will use a more balanced negative ratio and shorter, lower-augmentation training to reduce the v2 tendency to treat similar gray objects as cats.

## Fine-Tune v3: Balanced Hard Negatives

To avoid v2's cat-to-negative imbalance, 164 further sampled negative frames were extracted from the five hard-negative videos. The train split was capped at 80 negatives: 245 train frames (165 cat, 80 negative). Validation remains video-disjoint with 72 frames (47 cat, 25 negative).

Validation for `runs/p30_cat/yolo11n_p30_v3_balanced_negatives/weights/best.pt`:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.932 | 0.918 | 0.951 | 0.695 |

This is a candidate only. It must be tested on P30 with the bilinear native YUV path against the keyboard, cushions, wall picture, and lying cat. Threshold raising remains rejected because difficult true and false cases overlap in confidence.

## Targeted Capture Set

The P30 app captured nine exact runtime false-positive scenes: the wall artwork, keyboard/work desk, cushions, cat tower, and slippers. Review confirms no real cat appears in these frames. Fine-tune v4 uses each exact capture four times as an empty-label training image, alongside the v3 balanced dataset. This deliberately gives the actual failure modes more weight than generic empty rooms; it is an experiment and must still be rejected if it hurts lying-cat recall.

Validation for `runs/p30_cat/yolo11n_p30_v4_targeted_negatives/weights/best.pt`:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.951 | 0.898 | 0.947 | 0.702 |

The held-out set does not contain the exact captured scenes, so these metrics only clear the candidate for P30 runtime testing; they do not prove that keyboard/cushion/artwork false positives are fixed.

## Fine-Tune v5: DINO-Rescued Backlit Frames

18 backlit frames YOLO missed but Grounding DINO Tiny found (0.42–0.91)
were human-reviewed (17 keep, 1 redrawn) and merged. Train: 344 frames
(228 cat, 116 negative).

Validation for `runs/p30_cat/yolo11n_p30_v5_dino/weights/best.pt`:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.937 | 0.939 | 0.958 | 0.773 |

Recall up vs v4 (0.898), same mAP band. Exported to NCNN 384 + FP16
(`p30_cat_v5_384_fp16_ncnn_model`), Android wired, release APK built
and installed 2026-10-02.

P30 runtime (evening, no backlight available): wall picture,
cushions, AND keyboard show NO false positives; lying cat detected
OK; conf spans 0.3-0.9 across conditions. All testable gates pass —
v5 is accepted as the live model (already installed). Remaining:
daylight backlight retest to confirm the rescue frames translated
to runtime recall.

## P30 Runtime Rejection v3

P30 rejected v3 at about 100-120 ms / 8-9 FPS. Keyboard false-positive confidence rose to `0.6-0.8`; more cushions were detected at `0.5+`. The wall-picture patch was suppressed, but that isolated win does not offset the expanded keyboard/cushion false positives. Cat confidence remained broad (`0.4-0.9`). Do not train a v4 from uniformly sampled hard-negative video frames: they diluted rather than targeted the failures. Return to COCO and mine exact reviewed frames of the keyboard, false cushions, and picture patch before any further custom model experiment.
