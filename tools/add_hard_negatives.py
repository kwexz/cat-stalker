"""Extract confirmed no-cat P30 videos as empty-label hard negatives."""

import csv
import shutil
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_videos"
DATASET = ROOT / "data" / "candidate_frames"
PREFIX = "VID_20260929_115"
FRAMES_PER_VIDEO = 36


def main():
    manifest_path = DATASET / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as manifest:
        rows = list(csv.DictReader(manifest))
    existing = {row["image"] for row in rows}
    added = 0
    for video in sorted(RAW.glob(f"{PREFIX}*.mp4")):
        capture = cv2.VideoCapture(str(video))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        for index in sorted({round(i * (count - 1) / max(FRAMES_PER_VIDEO - 1, 1)) for i in range(FRAMES_PER_VIDEO)}):
            name = f"{video.stem}_f{index:06d}.jpg"
            if name in existing:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            cv2.imwrite(str(DATASET / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (DATASET / "labels" / f"{Path(name).stem}.txt").write_text("", encoding="ascii")
            rows.append({"image": name, "timestamp_s": f"{index / fps:.3f}", "bootstrap_cats": "0", "review": "negative"})
            added += 1
        capture.release()
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Added {added} confirmed hard-negative frames")


if __name__ == "__main__":
    main()
