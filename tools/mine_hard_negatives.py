"""Select the exact cat-free frames that a candidate model scores as cats."""

import csv
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "candidate_frames" / "images"
LABELS = ROOT / "data" / "candidate_frames" / "labels"
MODEL = ROOT / "runs" / "p30_cat" / "yolo11n_p30_v3_balanced_negatives" / "weights" / "best.pt"
OUTPUT = ROOT / "data" / "mined_hard_negatives"
PREFIX = "VID_20260929_115"
LIMIT = 40


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    detector = YOLO(str(MODEL))
    scored = []
    for image in sorted(IMAGES.glob(f"{PREFIX}*.jpg")):
        label = LABELS / f"{image.stem}.txt"
        if label.read_text(encoding="ascii").strip():
            continue
        result = detector(str(image), imgsz=480, conf=0.01, verbose=False)[0]
        confidence = max((float(box.conf.item()) for box in result.boxes), default=0.0)
        scored.append((confidence, image))
    scored.sort(reverse=True, key=lambda item: item[0])
    rows = []
    for confidence, image in scored[:LIMIT]:
        shutil.copy2(image, OUTPUT / "images" / image.name)
        rows.append({"image": image.name, "v3_confidence": f"{confidence:.4f}", "label": "negative"})
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Mined {len(rows)} false-positive candidates; top confidence={rows[0]['v3_confidence']}")


if __name__ == "__main__":
    main()
