"""Compare PyTorch and NCNN outputs for the P30 fine-tune on identical pixels."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import ncnn
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "runs" / "p30_cat" / "yolo11n_p30_v2_hard_negatives" / "weights" / "best.pt"
NCNN = ROOT / "runs" / "p30_cat" / "yolo11n_p30_v2_hard_negatives" / "weights" / "best_ncnn_model"
NCNN_FP16 = ROOT / "android" / "app" / "src" / "main" / "assets" / "p30_cat_v2_384_fp16_ncnn_model"
IMAGE = ROOT / "data" / "candidate_frames" / "images" / "VID_20260929_084341_f000000.jpg"
SIZE = 384


def letterbox(image: np.ndarray):
    height, width = image.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    top = (SIZE - resized.shape[0]) // 2
    left = (SIZE - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def pytorch_output(image: np.ndarray):
    model = YOLO(str(MODEL)).model.eval()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        return model(tensor)[0].detach().numpy()[0]


def ncnn_output(image: np.ndarray, model_dir: Path):
    net = ncnn.Net()
    net.load_param(str(model_dir / "model.ncnn.param"))
    net.load_model(str(model_dir / "model.ncnn.bin"))
    mat = ncnn.Mat.from_pixels(image, ncnn.Mat.PixelType.PIXEL_BGR2RGB, SIZE, SIZE)
    mat.substract_mean_normalize((), (1 / 255.0, 1 / 255.0, 1 / 255.0))
    extractor = net.create_extractor()
    extractor.input("in0", mat)
    _, output = extractor.extract("out0")
    return np.array(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=IMAGE)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot read {args.image}")
    input_image = letterbox(image)
    torch_output = pytorch_output(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ncnn_result = ncnn_output(input_image, NCNN)
    fp16_result = ncnn_output(input_image, NCNN_FP16)
    print(f"input={args.image.name} torch={torch_output.shape} ncnn={ncnn_result.shape} fp16={fp16_result.shape}")
    print(f"torch max confidence={torch_output[4].max():.6f}")
    print(f"ncnn max confidence={ncnn_result[4].max():.6f}")
    print(f"fp16 max confidence={fp16_result[4].max():.6f}")
    print(f"max abs difference={np.abs(torch_output - ncnn_result).max():.6f}")
    print(f"fp16 max abs difference={np.abs(ncnn_result - fp16_result).max():.6f}")


if __name__ == "__main__":
    main()
