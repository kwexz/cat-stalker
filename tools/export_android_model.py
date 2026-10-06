"""Export the Cat Stalker YOLO11n detector as a float32 TFLite asset.

Run this on Linux x86_64 or macOS: current Ultralytics LiteRT export does not
support Windows hosts.
"""

from pathlib import Path
import shutil
import platform
import argparse

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = ROOT / "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--ncnn-only", action="store_true")
    args = parser.parse_args()
    if platform.system() == "Windows":
        raise SystemExit(
            "Ultralytics LiteRT/TFLite export is unsupported on Windows. "
            "Run this command in WSL2 Ubuntu or on Linux/macOS."
        )
    if not SOURCE_MODEL.is_file():
        raise SystemExit(f"Missing source model: {SOURCE_MODEL}")

    asset_dir = ROOT / "android" / "app" / "src" / "main" / "assets"
    if not args.ncnn_only:
        exported = Path(YOLO(str(SOURCE_MODEL)).export(format="litert", imgsz=args.imgsz, nms=False, int8=False))
        if not exported.is_file():
            raise SystemExit(f"Ultralytics did not create: {exported}")
        output_model = asset_dir / f"yolo11n_{args.imgsz}_float32.tflite"
        shutil.copy2(exported, output_model)
        print(f"Copied {exported.name} to {output_model}")

    ncnn_dir = Path(YOLO(str(SOURCE_MODEL)).export(format="ncnn", imgsz=args.imgsz, simplify=True))
    ncnn_output_dir = asset_dir / f"yolo11n_{args.imgsz}_ncnn_model"
    ncnn_output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.ncnn.param", "model.ncnn.bin"):
        shutil.copy2(ncnn_dir / name, ncnn_output_dir / name)
    print(f"Copied NCNN model to {ncnn_output_dir}")


if __name__ == "__main__":
    main()
