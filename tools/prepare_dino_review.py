"""DINO-assisted review set for v5: frames YOLO misses but DINO finds.

Scans imported session frames, keeps YOLO<0.28 & DINO>=0.35 (plus
DINO>=0.5 borderline for context), writes DINO boxes as provisional
labels into data/dino_review for the keyboard review tool.
"""

import csv
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "candidate_frames"
OUTPUT = ROOT / "data" / "dino_review"
YOLO_THRESH = 0.28
DINO_THRESH = 0.35


def main():
    import importlib.util
    from ultralytics import YOLO
    spec = importlib.util.spec_from_file_location(
        "label_grounding_dino", ROOT / "tools" / "label_grounding_dino.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load_detector, detect, DEFAULT_MODEL = mod.load_detector, mod.detect, mod.DEFAULT_MODEL

    session_images = sorted(DATASET.glob("images/179*.jpg"))
    print(f"session frames: {len(session_images)}")
    yolo = YOLO(str(ROOT / "yolo11n.pt"))
    processor, dino = load_detector(DEFAULT_MODEL)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    (OUTPUT / "labels").mkdir()

    rows = []
    for path in session_images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        yolo_best = 0.0
        result = yolo(str(path), imgsz=480, conf=0.15, verbose=False)[0]
        for box in result.boxes:
            if int(box.cls.item()) == 15:
                yolo_best = max(yolo_best, float(box.conf.item()))
        dino_boxes = detect(processor, dino, image, "cat", 0.25, 0.25)
        dino_best = max([d["score"] for d in dino_boxes], default=0.0)
        if not (yolo_best < YOLO_THRESH and dino_best >= DINO_THRESH):
            continue
        shutil.copy2(path, OUTPUT / "images" / path.name)
        h, w = image.shape[:2]
        with (OUTPUT / "labels" / f"{path.stem}.txt").open("w", encoding="ascii") as f:
            for d in dino_boxes:
                x1, y1, x2, y2 = d["box"]
                f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        rows.append({"image": path.name, "yolo": f"{yolo_best:.3f}",
                     "dino": f"{dino_best:.3f}", "decision": "TODO", "note": ""})
        print(f"rescue {path.name} yolo={yolo_best:.2f} dino={dino_best:.2f}")
    with (OUTPUT / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rescue set: {len(rows)} frames -> {OUTPUT}")


if __name__ == "__main__":
    main()
