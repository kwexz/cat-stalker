"""Copy reviewed compact labels back into the full candidate set."""

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "candidate_frames" / "labels"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="label_review")
    args = parser.parse_args()
    REVIEW = ROOT / "data" / args.dir
    with (REVIEW / "review.csv").open(newline="", encoding="utf-8") as review:
        rows = list(csv.DictReader(review))
    incomplete = [row["image"] for row in rows if row["decision"] in {"TODO", "skip"}]
    if incomplete:
        raise SystemExit(f"Review incomplete: {len(incomplete)} frames remain")
    for row in rows:
        name = f"{Path(row['image']).stem}.txt"
        shutil.copy2(REVIEW / "labels" / name, SOURCE / name)
    print(f"Applied {len(rows)} reviewed labels to {SOURCE}")


if __name__ == "__main__":
    main()
