"""Import phone session event frames into the candidate dataset.

Copies JPEGs, writes YOLO bootstrap labels (same format as
prepare_cat_dataset.py), appends manifest rows with review=TODO.
Idempotent: skips images already present.
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "candidate_frames"
MODEL = ROOT / "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    frames = sorted((ROOT / args.session / "frames").glob("*.jpg"))
    detector = YOLO(str(MODEL))
    manifest_path = DATASET / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    existing = {r["image"] for r in rows}
    added = 0
    for frame in frames:
        if frame.name in existing:
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        dest = DATASET / "images" / frame.name
        shutil.copy2(frame, dest)
        result = detector(str(dest), imgsz=480, conf=args.conf, verbose=False)[0]
        label_path = DATASET / "labels" / f"{dest.stem}.txt"
        cats = 0
        with label_path.open("w", encoding="ascii") as labels:
            for box in result.boxes:
                if int(box.cls.item()) != 15:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h, w = result.orig_shape
                labels.write(
                    f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
                )
                cats += 1
        rows.append({
            "image": dest.name,
            "timestamp_s": "0.000",
            "bootstrap_cats": str(cats),
            "review": "TODO",
        })
        added += 1
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {added} session frames into {DATASET}")


if __name__ == "__main__":
    main()
