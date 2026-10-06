"""Extract diverse P30 video frames and bootstrap cat labels for review."""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RAW_VIDEOS = ROOT / "data" / "raw_videos"
OUTPUT = ROOT / "data" / "candidate_frames"
MODEL = ROOT / "yolo11n.pt"
CAT_CLASS = 15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--negative-video-prefix", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def frame_name(video: Path, index: int) -> str:
    return f"{video.stem}_f{index:06d}.jpg"


def extract_frames(video: Path, frames_per_video: int) -> list[tuple[Path, float]]:
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"Cannot read video metadata: {video}")

    selected = []
    for index in sorted({round(i * (frame_count - 1) / max(frames_per_video - 1, 1)) for i in range(frames_per_video)}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        destination = OUTPUT / "images" / frame_name(video, index)
        cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        selected.append((destination, index / fps))
    capture.release()
    return selected


def bootstrap_labels(images: list[tuple[Path, float]], negative_prefix: str) -> list[dict[str, str]]:
    detector = YOLO(str(MODEL))
    rows = []
    for image, timestamp in images:
        is_negative_video = negative_prefix and image.name.startswith(negative_prefix)
        result = None if is_negative_video else detector(str(image), imgsz=480, conf=0.15, verbose=False)[0]
        label_path = OUTPUT / "labels" / f"{image.stem}.txt"
        cat_count = 0
        with label_path.open("w", encoding="ascii") as labels:
            for box in ([] if result is None else result.boxes):
                if int(box.cls.item()) != CAT_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                height, width = result.orig_shape
                x = (x1 + x2) / 2 / width
                y = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                labels.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                cat_count += 1
        rows.append({"image": image.name, "timestamp_s": f"{timestamp:.3f}", "bootstrap_cats": str(cat_count), "review": "negative" if is_negative_video else "TODO"})
    return rows


def main():
    args = parse_args()
    videos = sorted(RAW_VIDEOS.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4 videos in {RAW_VIDEOS}")
    if OUTPUT.exists() and not args.overwrite:
        raise SystemExit(f"{OUTPUT} already exists; use --overwrite to replace it")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    (OUTPUT / "labels").mkdir()

    extracted = []
    for video in videos:
        extracted.extend(extract_frames(video, args.frames_per_video))
    rows = bootstrap_labels(extracted, args.negative_video_prefix)
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extracted {len(extracted)} frames from {len(videos)} videos into {OUTPUT}")


if __name__ == "__main__":
    main()
