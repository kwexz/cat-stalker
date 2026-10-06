"""Build a video-disjoint single-class YOLO dataset from reviewed P30 frames."""

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "candidate_frames"
REVIEW = ROOT / "data" / "label_review" / "review.csv"
OUTPUT = ROOT / "data" / "p30_cat_yolo"
TARGETED_NEGATIVES = ROOT / "data" / "hard_negative_captures"
VALIDATION_VIDEOS = {
    "VID_20260929_084704",
    "VID_20260929_084738",
    "VID_20260929_115659",
    "VID_20260929_115710",
}


def source_video(name: str) -> str:
    return "_".join(name.split("_")[:3])


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for split in ("train", "val"):
        (OUTPUT / "images" / split).mkdir(parents=True)
        (OUTPUT / "labels" / split).mkdir(parents=True)

    reviewed = set()
    with REVIEW.open(newline="", encoding="utf-8") as review:
        for row in csv.DictReader(review):
            if row["decision"] not in {"keep", "draw", "remove"}:
                raise SystemExit(f"Incomplete review: {row['image']}")
            reviewed.add(row["image"])

    copied = {"train": 0, "val": 0}
    positives = {"train": 0, "val": 0}
    negative_limit = {"train": 80, "val": 25}
    for image in sorted((SOURCE / "images").glob("*.jpg")):
        video = source_video(image.name)
        split = "val" if video in VALIDATION_VIDEOS else "train"
        label = SOURCE / "labels" / f"{image.stem}.txt"
        is_negative = not label.read_text(encoding="ascii").strip()
        if is_negative and copied[split] - positives[split] >= negative_limit[split]:
            continue
        shutil.copy2(image, OUTPUT / "images" / split / image.name)
        shutil.copy2(label, OUTPUT / "labels" / split / label.name)
        copied[split] += 1
        positives[split] += bool(label.read_text(encoding="ascii").strip())

    targeted = sorted(TARGETED_NEGATIVES.glob("CAT_STALKER_NEG_*.jpg"))
    if targeted:
        for image in targeted:
            for repeat in range(4):
                destination = OUTPUT / "images" / "train" / f"targeted_{image.stem}_{repeat}.jpg"
                shutil.copy2(image, destination)
                (OUTPUT / "labels" / "train" / f"{destination.stem}.txt").write_text("", encoding="ascii")
                copied["train"] += 1

    (OUTPUT / "data.yaml").write_text(
        f"path: {OUTPUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n",
        encoding="ascii",
    )
    print(
        f"Built dataset: train={copied['train']} ({positives['train']} cat, {copied['train'] - positives['train']} negative, "
        f"targeted_captures={len(targeted)}) val={copied['val']} ({positives['val']} cat, {copied['val'] - positives['val']} negative) reviewed={len(reviewed)}"
    )


if __name__ == "__main__":
    main()
