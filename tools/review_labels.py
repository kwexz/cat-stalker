"""Minimal keyboard review for provisional P30 cat labels."""

import csv
import os
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def parse_review_dir():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.environ.get("REVIEW_DIR", "label_review"))
    args, _ = parser.parse_known_args()
    review = ROOT / "data" / args.dir
    if not (review / "review.csv").exists():
        sys.exit(f"No review.csv in {review} (use --dir <name>)")
    return review


REVIEW = parse_review_dir()


def read_box(path: Path):
    lines = path.read_text(encoding="ascii").splitlines()
    return None if not lines else list(map(float, lines[0].split()[1:]))


def normalize_key(raw):
    """Accept Latin and Russian layouts: physical Y/N/M/S/Q keys."""
    mapping = {
        ord("y"): "y", ord("Y"): "y", ord("н"): "y", ord("Н"): "y",
        ord("n"): "n", ord("N"): "n", ord("т"): "n", ord("Т"): "n",
        ord("m"): "m", ord("M"): "m", ord("ь"): "m", ord("Ь"): "m",
        ord("s"): "s", ord("S"): "s", ord("ы"): "s", ord("Ы"): "s",
        ord("q"): "q", ord("Q"): "q", ord("й"): "q", ord("Й"): "q",
    }
    return mapping.get(raw, "")


def draw(image, box):
    if box is None:
        return image
    height, width = image.shape[:2]
    x, y, w, h = box
    cv2.rectangle(image, (int((x - w / 2) * width), int((y - h / 2) * height)), (int((x + w / 2) * width), int((y + h / 2) * height)), (0, 220, 255), 3)
    return image


class BoxEditor:
    def __init__(self, image, box):
        self.image = image
        self.box = box
        self.start = None
        self.drawing = False

    def mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x1, y1 = self.start
            x1, x2 = sorted((x1, x))
            y1, y2 = sorted((y1, y))
            height, width = self.image.shape[:2]
            if x2 > x1 and y2 > y1:
                self.box = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
            self.drawing = False

    def preview(self):
        return draw(self.image.copy(), self.box)


def write_box(path: Path, box):
    path.write_text("" if box is None else "0 %.6f %.6f %.6f %.6f\n" % tuple(box), encoding="ascii")


def main():
    with (REVIEW / "review.csv").open(newline="", encoding="utf-8") as review:
        rows = list(csv.DictReader(review))
    pending = sum(1 for r in rows if r["decision"] == "TODO")
    print(f"{REVIEW.name}: {pending}/{len(rows)} frames pending review")
    if not pending:
        print("Nothing to review — all frames already decided.")
        return
    for row in rows:
        if row["decision"] != "TODO":
            continue
        image_path = REVIEW / "images" / row["image"]
        label_path = REVIEW / "labels" / f"{image_path.stem}.txt"
        editor = BoxEditor(cv2.imread(str(image_path)), read_box(label_path))
        window = "P30 Cat Label Review"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)
        cv2.setMouseCallback(window, editor.mouse)
        while True:
            image = editor.preview()
            cv2.putText(image, "Y keep | N no cat | M draw/replace box | S skip | Q save and quit (RU layout OK)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window, image)
            raw = cv2.waitKey(20)
            if raw < 0:
                continue
            key = normalize_key(raw)
            if not key:
                continue
            break
        if key == "q":
            break
        if key == "y":
            write_box(label_path, editor.box)
            row["decision"] = "keep"
        elif key == "n":
            write_box(label_path, None)
            row["decision"] = "remove"
        elif key == "m":
            if editor.box is None:
                row["note"] = "draw a box before pressing M"
                continue
            write_box(label_path, editor.box)
            row["decision"] = "draw"
        elif key == "s":
            row["decision"] = "skip"
        else:
            continue
        with (REVIEW / "review.csv").open("w", newline="", encoding="utf-8") as review:
            writer = csv.DictWriter(review, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
