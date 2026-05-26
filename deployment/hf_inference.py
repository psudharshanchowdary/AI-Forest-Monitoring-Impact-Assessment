from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

CLASS_COLORS = {
    0: (34, 197, 94),
    1: (250, 204, 21),
    2: (59, 130, 246),
    3: (249, 115, 22),
    4: (239, 68, 68),
}


class ForestSegmentationInference:
    def __init__(self, checkpoint: str | Path | None = None, model_repo_id: str | None = None) -> None:
        if checkpoint is None:
            if model_repo_id is None:
                raise ValueError("Either checkpoint or model_repo_id must be provided")
            checkpoint = hf_hub_download(repo_id=model_repo_id, filename="best.pt")
        self.model = YOLO(str(checkpoint))

    def predict(self, image: np.ndarray, imgsz: int = 640, conf: float = 0.25) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image array")
        results = self.model.predict(source=image, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
        overlay = image.copy().astype(np.float32)
        records: list[dict[str, Any]] = []
        if results.masks is None or results.boxes is None:
            return overlay.astype(np.uint8), records

        masks = results.masks.data.detach().cpu().numpy()
        labels = results.boxes.cls.detach().cpu().numpy().astype(int)
        scores = results.boxes.conf.detach().cpu().numpy().astype(float)
        names = results.names
        for idx, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            binary = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            color = np.asarray(CLASS_COLORS.get(label, (255, 255, 255)), dtype=np.float32)
            overlay[binary] = (0.60 * overlay[binary]) + (0.40 * color)
            records.append({
                "instance_id": idx,
                "class_name": names.get(label, str(label)),
                "confidence": round(float(score), 4),
                "pixel_area": int(binary.sum()),
            })
        return np.clip(overlay, 0, 255).astype(np.uint8), records
