"""Offline candidate shootout on P30 hard frames. No training, just numbers.

Compares COCO baseline, our tuned models, and external animal detectors
on the exact failure modes: backlit cats, normal cats, hard negatives.
A candidate that wins here still needs an NCNN export + P30 runtime
check before replacing anything on the phone.
"""

import csv
import glob
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
HUB = Path.home() / ".cache/huggingface/hub/models--bsgcasa--yolov8-animal-detector"

MODELS = {
    "coco-yolo11n": (ROOT / "yolo11n.pt", 15),
    "p30-v4": (
        ROOT / "runs/p30_cat/yolo11n_p30_v4_targeted_negatives/weights/best.pt",
        0,
    ),
    "bsgcasa-yolov8": (next(HUB.glob("snapshots/*/best.pt")), 2),
}

CONF = 0.28


def collect():
    backlit = sorted(
        p for p in Path(ROOT / "sessions/20261002_141914/frames").glob("*.jpg")
        if "_confna" not in p.name
    )[:12]
    positives = []
    manifest = ROOT / "data/candidate_frames/manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["bootstrap_cats"]) > 0:
                    positives.append(ROOT / "data/candidate_frames/images" / row["image"])
                if len(positives) >= 12:
                    break
    negatives = sorted((ROOT / "data/hard_negative_captures").glob("*.jpg"))
    return {"backlit": backlit, "positive": positives, "negative": negatives}


def main():
    sets = collect()
    print({k: len(v) for k, v in sets.items()})
    for name, (weights, cat_cls) in MODELS.items():
        if not Path(weights).exists():
            print(f"{name}: MISSING {weights}")
            continue
        model = YOLO(str(weights))
        print(f"== {name} (cat class {cat_cls}) ==")
        for set_name, paths in sets.items():
            dets, confs = 0, []
            for path in paths:
                result = model(str(path), imgsz=480, conf=CONF, verbose=False)[0]
                best = 0.0
                for box in result.boxes:
                    if int(box.cls.item()) == cat_cls:
                        best = max(best, float(box.conf.item()))
                if best > 0:
                    dets += 1
                    confs.append(best)
            n = max(len(paths), 1)
            mean_conf = sum(confs) / max(len(confs), 1)
            print(f"  {set_name:8s} {dets:3d}/{len(paths):3d} mean_conf={mean_conf:.2f}")


if __name__ == "__main__":
    main()
