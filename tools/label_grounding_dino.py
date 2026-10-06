"""Grounding DINO Tiny as PC-side cat labeler (NOT a phone runtime).

Compares open-vocabulary boxes against the YOLO bootstrap on the same
P30 frames: the question is recall lift for dataset labeling, especially
backlit/lying cats YOLO misses. Writes dino_labels.json + comparison.
"""

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def load_detector(model_id):
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()
    return processor, model


def detect(processor, model, image_bgr, prompt, box_threshold, text_threshold):
    import torch

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, text=[[prompt]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[image_rgb.shape[:2]],
        threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]
    out = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        out.append({"box": [round(float(v), 1) for v in box.tolist()],
                    "score": round(float(score), 4), "label": label})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--prompt", default="cat")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/dino_labels.json")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    processor, model = load_detector(args.model)
    results = {}
    for pattern in args.images:
        for path in sorted(Path(ROOT).glob(pattern)):
            if args.max_images and len(results) >= args.max_images:
                break
            image = cv2.imread(str(path))
            if image is None:
                continue
            results[path.name] = detect(
                processor, model, image, args.prompt,
                args.box_threshold, args.text_threshold,
            )
            best = max([d["score"] for d in results[path.name]], default=0.0)
            print(f"{path.name}: {len(results[path.name])} boxes, best={best:.3f}")
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"Wrote {len(results)} images -> {out_path}")


if __name__ == "__main__":
    main()
