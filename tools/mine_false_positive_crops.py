"""Mine high-confidence v3 false-positive crops from confirmed cat-free frames."""

import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "candidate_frames" / "images"
LABELS = ROOT / "data" / "candidate_frames" / "labels"
MODEL = ROOT / "runs" / "p30_cat" / "yolo11n_p30_v3_balanced_negatives" / "weights" / "best.pt"
OUTPUT = ROOT / "data" / "mined_false_positive_crops"
PREFIX = "VID_20260929_115"
CONFIDENCE = 0.05
LIMIT = 30


def crop_with_context(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    height, width = image.shape[:2]
    side = max(x2 - x1, y2 - y1, 48)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    left = max(0, center_x - side)
    top = max(0, center_y - side)
    right = min(width, center_x + side)
    bottom = min(height, center_y + side)
    return image[top:bottom, left:right], (x1, y1, x2, y2)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    detector = YOLO(str(MODEL))
    candidates = []
    for path in sorted(IMAGES.glob(f"{PREFIX}*.jpg")):
        if LABELS.joinpath(f"{path.stem}.txt").read_text(encoding="ascii").strip():
            continue
        image = cv2.imread(str(path))
        result = detector(str(path), imgsz=480, conf=CONFIDENCE, verbose=False)[0]
        for index, box in enumerate(result.boxes):
            crop, xyxy = crop_with_context(image, box)
            if crop.size:
                candidates.append((float(box.conf.item()), path, index, crop, xyxy))
    candidates.sort(reverse=True, key=lambda item: item[0])
    rows = []
    for rank, (confidence, source, index, crop, xyxy) in enumerate(candidates[:LIMIT], start=1):
        name = f"fp_{rank:02d}_{source.stem}_b{index}.jpg"
        cv2.imwrite(str(OUTPUT / "images" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"image": name, "source": source.name, "confidence": f"{confidence:.4f}", "source_box": " ".join(map(str, xyxy)), "review": "TODO"})
    with (OUTPUT / "review.csv").open("w", newline="", encoding="utf-8") as review:
        writer = csv.DictWriter(review, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Mined {len(rows)} false-positive crops from {len(candidates)} candidate boxes")


if __name__ == "__main__":
    main()
