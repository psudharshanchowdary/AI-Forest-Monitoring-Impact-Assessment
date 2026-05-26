from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 7 ONNX export and benchmark for YOLOv8 segmentation")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "yolov8_seg" / "weights" / "best.pt")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "onnx_benchmark")
    return parser.parse_args()



def load_tensors(image_paths: list[Path], imgsz: int) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    torch_inputs: list[torch.Tensor] = []
    onnx_inputs: list[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        array = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        torch_inputs.append(tensor)
        onnx_inputs.append(tensor.numpy())
    return torch_inputs, onnx_inputs



def benchmark_pytorch(model, inputs: list[torch.Tensor], device: torch.device) -> float:
    model.eval()
    timings = []
    with torch.no_grad():
        for tensor in inputs:
            tensor = tensor.to(device)
            start = time.perf_counter()
            _ = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings))



def benchmark_onnx(session: ort.InferenceSession, inputs: list[np.ndarray]) -> float:
    input_name = session.get_inputs()[0].name
    timings = []
    for array in inputs:
        start = time.perf_counter()
        _ = session.run(None, {input_name: array})
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings))



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(list((args.dataset_root / args.split / "images").glob("*.png")) + list((args.dataset_root / args.split / "images").glob("*.jpg")))[: args.samples]
    if not image_paths:
        raise RuntimeError(f"No images found under {args.dataset_root / args.split / 'images'}")

    yolo = YOLO(str(args.checkpoint))
    export_result = yolo.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False)
    onnx_path = Path(str(export_result))

    torch_inputs, onnx_inputs = load_tensors(image_paths, imgsz=args.imgsz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model = yolo.model.to(device)

    pytorch_ms = benchmark_pytorch(pt_model, torch_inputs, device=device)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_ms = benchmark_onnx(session, onnx_inputs)
    speed_improvement = ((pytorch_ms - onnx_ms) / max(pytorch_ms, 1e-6)) * 100.0

    payload = {
        "checkpoint": str(args.checkpoint),
        "onnx_path": str(onnx_path),
        "pytorch_ms_per_image": pytorch_ms,
        "onnx_ms_per_image": onnx_ms,
        "speed_improvement_percent": speed_improvement,
        "sample_count": len(torch_inputs),
    }
    (args.output_dir / "speed_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    plt.bar(["PyTorch", "ONNX Runtime"], [pytorch_ms, onnx_ms], color=["#2563eb", "#16a34a"])
    plt.ylabel("Latency (ms/image)")
    plt.title("YOLOv8 Segmentation Inference Speed")
    plt.tight_layout()
    plt.savefig(args.output_dir / "speed_comparison.png", dpi=180)
    plt.close()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
