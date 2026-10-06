"""Select a compact, diverse review set from bootstrap cat labels."""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "candidate_frames"
OUTPUT = ROOT / "data" / "label_review"
MAX_PER_VIDEO = 6


def load_box(path: Path):
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        return None
    _, x, y, width, height = map(float, lines[0].split())
    return x, y, width, height


def score(image: Path, box):
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    contrast = float(gray.std()) if gray is not None else 0.0
    if box is None:
        return 1_000_000.0 + contrast
    _, _, width, height = box
    area = width * height
    distance_from_center = abs(box[0] - 0.5) + abs(box[1] - 0.5)
    difficulty = 1.0 / max(area, 0.0001)
    return difficulty * 100.0 + distance_from_center * 10.0 + contrast / 100.0


def source_video(name: str) -> str:
    return "_".join(name.split("_")[:3])


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    (OUTPUT / "labels").mkdir()

    groups = defaultdict(list)
    for image in sorted((SOURCE / "images").glob("*.jpg")):
        label = SOURCE / "labels" / f"{image.stem}.txt"
        box = load_box(label)
        groups[source_video(image.name)].append((score(image, box), image, label, box))

    selected = []
    for video, items in sorted(groups.items()):
        items.sort(key=lambda item: item[0], reverse=True)
        selected.extend((video, *item) for item in items[:MAX_PER_VIDEO])

    rows = []
    for video, _, image, label, box in selected:
        shutil.copy2(image, OUTPUT / "images" / image.name)
        shutil.copy2(label, OUTPUT / "labels" / label.name)
        rows.append({
            "image": image.name,
            "source_video": video,
            "bootstrap": "miss" if box is None else "box",
            "decision": "TODO",
            "note": "",
        })
    with (OUTPUT / "review.csv").open("w", newline="", encoding="utf-8") as review:
        writer = csv.DictWriter(review, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {len(rows)} review images from {len(groups)} videos into {OUTPUT}")


if __name__ == "__main__":
    main()
