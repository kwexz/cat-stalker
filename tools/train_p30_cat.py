"""Fine-tune YOLO11n on the reviewed P30 cat dataset."""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "p30_cat_yolo" / "data.yaml"
RUNS = ROOT / "runs" / "p30_cat"


def main():
    model = YOLO(str(ROOT / "yolo11n.pt"))
    model.train(
        data=str(DATA),
        epochs=30,
        imgsz=480,
        batch=8,
        device="cpu",
        workers=0,
        patience=8,
        project=str(RUNS),
        name="yolo11n_p30_v5_dino",
        exist_ok=True,
        pretrained=True,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.1,
    )


if __name__ == "__main__":
    main()
