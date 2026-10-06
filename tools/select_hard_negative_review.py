"""Build a small manual review set from P30 hard-negative frames."""

import shutil
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "candidate_frames"
OUTPUT = ROOT / "data" / "hard_negative_review"
PREFIX = "VID_20260929_115"
PER_VIDEO = 8


def video_key(path: Path) -> str:
    return "_".join(path.stem.split("_")[:3])


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    (OUTPUT / "labels").mkdir()
    images = sorted((SOURCE / "images").glob(f"{PREFIX}*.jpg"))
    groups = {}
    for image in images:
        groups.setdefault(video_key(image), []).append(image)
    selected = []
    for _, group in sorted(groups.items()):
        for index in {round(i * (len(group) - 1) / max(PER_VIDEO - 1, 1)) for i in range(PER_VIDEO)}:
            selected.append(group[index])
    for image in selected:
        shutil.copy2(image, OUTPUT / "images" / image.name)
        (OUTPUT / "labels" / f"{image.stem}.txt").write_text("", encoding="ascii")
    cols, width, height = 5, 320, 205
    canvas = __import__("numpy").full((((len(selected) + cols - 1) // cols) * height, cols * width, 3), 32, dtype="uint8")
    for index, image in enumerate(selected):
        frame = cv2.imread(str(image))
        row, column = divmod(index, cols)
        canvas[row * height:row * height + 180, column * width:column * width + width] = cv2.resize(frame, (width, 180))
        cv2.putText(canvas, image.name[-15:-4], (column * width + 5, row * height + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.imwrite(str(OUTPUT / "contact_sheet.jpg"), canvas)
    print(f"Prepared {len(selected)} exact hard-negative review frames in {OUTPUT}")


if __name__ == "__main__":
    main()
