# P30 Cat Dataset Workflow

## What Is Collected

The source videos come from the Huawei P30 rear camera in the apartment. They contain the real cat across lying, sitting, walking, close-up, partially occluded, daylight, and indoor scenes.

Raw videos are stored locally in `data/raw_videos/` and ignored by git. Extracted review frames are stored locally in `data/candidate_frames/` and are also ignored by git.

## Current Extraction

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\prepare_cat_dataset.py --overwrite
```

This extracts 24 uniformly spaced frames from every MP4 and uses the current `yolo11n.pt` to create provisional `cat` labels in YOLO format. It does not create training-ready truth: every provisional label must be reviewed.

The generated paths are:

```text
data/candidate_frames/images/
data/candidate_frames/labels/
data/candidate_frames/manifest.csv
data/candidate_frames/contact_sheet.jpg
```

## Review Rules

- Keep a cat box only when it encloses the visible body accurately enough for detection training.
- Add a box if the real cat is visible but the bootstrap detector missed it.
- Delete boxes on reflections, printed animals, cushions, or unrelated objects.
- Leave images with no cat as valid negative images; their label `.txt` file remains empty.
- Prefer diversity over near-duplicate frames. Keep difficult distant and lying-cat frames even when the box is small.

## Exact Review Step

The tooling has already reduced the nine videos to 54 difficult or diverse frames in `data/label_review/`. This is the only manual contribution needed before a first fine-tune.

```powershell
.\.venv\Scripts\python.exe tools\review_labels.py
```

For every frame, use exactly one action:

- `Y`: the displayed box is correct; keep it.
- `N`: no real cat is in the frame; remove a false box or keep the frame as a negative.
- `M`: draw a tight box by dragging with the left mouse button, then press `M` to save it. Use this for a missed cat or to replace a poor box.
- `S`: leave an uncertain frame for later; avoid this unless truly unsure.
- `Q`: save all completed decisions and stop.

After all 54 frames are decided, run:

```powershell
.\.venv\Scripts\python.exe tools\apply_label_review.py
```

## Split Before Training

Split by source video, not by individual frame. A frame from one video must never appear in both train and validation, otherwise validation is artificially optimistic.

Suggested initial split: seven source videos for training and two complete source videos for validation. Do not train until reviewed labels are available.

The first retraining iteration adds 60 explicitly cat-free hard-negative frames. Two complete hard-negative videos are held out with the cat validation videos, so false-positive behavior is measured rather than assumed.

## First Fine-Tune

After applying review labels, build the video-disjoint dataset and train:

```powershell
.\.venv\Scripts\python.exe tools\build_p30_dataset.py
.\.venv\Scripts\python.exe tools\train_p30_cat.py
```

The current Windows workstation has no CUDA GPU, so the first run uses CPU. It is intentionally a small 60-epoch YOLO11n fine-tune with conservative geometric augmentation; the target camera is mounted low and its orientation should not be randomized aggressively.
