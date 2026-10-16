# Third-party software, models, and data

The root [`LICENSE`](LICENSE) applies to original Cat Stalker source code and documentation unless a file says otherwise. It does **not** relicense third-party libraries, pretrained/fine-tuned model artifacts, datasets, Android components, or media.

This file is an inventory and review checklist, not legal advice and not a substitute for upstream license texts.

## Ultralytics YOLO and model weights

The project uses the `ultralytics` Python package, YOLO11 architecture/tooling, a COCO-pretrained `yolo11n.pt` base, and exported TFLite/NCNN artifacts. The Android asset directory currently includes a TFLite baseline and an active fine-tuned NCNN model.

Ultralytics publishes AGPL-3.0 and Enterprise licensing options. Determine which terms apply to the exact use—especially redistribution of weights, embedding in an APK, network services, modification, and commercial deployment—before distributing an application or model bundle.

- Licensing overview: <https://www.ultralytics.com/license>
- Upstream repository license: <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- YOLO11 documentation: <https://docs.ultralytics.com/models/yolo11/>

Do not infer that exported or fine-tuned weights become MIT merely because this repository's original code is MIT.

## COCO

The base model was trained using COCO. No COCO images are intentionally included in this repository. COCO images and annotations have their own source licenses/terms and must be assessed separately if downloaded or redistributed.

- Terms of use: <https://cocodataset.org/#termsofuse>

## Local Huawei P30 dataset and project media

The custom training material consists of private recordings of the author's cat and home. Raw videos, extracted frames, event images, and training image trees are intentionally excluded from git. Text labels/manifests and aggregate result reports are retained for engineering traceability.

No broad dataset license is granted by the root MIT license. The tracked setup
photo under `docs/` is project media, not a training dataset grant. Obtain
explicit permission before republishing personal imagery beyond normal
repository display/fork behavior.

Fine-tuned weights are derivative artifacts with both training-data and base-model considerations. They require a separate redistribution decision.

## Runtime and build dependencies

The project also depends on or interfaces with:

- Tencent NCNN;
- Android SDK, AndroidX, CameraX, Kotlin, Gradle, CMake, and the Android NDK;
- OpenCV;
- pygame;
- `python-miio` and its dependencies;
- ByteTrack integration supplied through Ultralytics;
- optional research tooling such as Grounding DINO and PyTorch used during dataset experiments.

Each retains its upstream license and notice requirements. The NCNN fetch script downloads a versioned upstream binary release, but a release/distribution process should pin hashes and collect applicable notices rather than relying only on this development helper.

## Distribution checklist

Before publishing an APK, wheel, executable, container, model archive, or dataset:

1. enumerate every bundled dependency and artifact;
2. preserve required copyright/license/notice files;
3. verify Ultralytics/base-weight terms for the intended distribution and deployment model;
4. verify rights to all training and project media;
5. document model provenance, version, format, and checksums;
6. do not describe the whole bundle as “MIT licensed” when only the original project code is MIT.
