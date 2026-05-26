"""NDVI utilities for training and inference workflows."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    if float(np.nanmax(red)) > 1.5 or float(np.nanmax(nir)) > 1.5:
        red = red / 10000.0
        nir = nir / 10000.0

    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.clip(ndvi, -1.0, 1.0)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)

    if float(np.nanmax(green)) > 1.5 or float(np.nanmax(nir)) > 1.5:
        green = green / 10000.0
        nir = nir / 10000.0

    ndwi = (green - nir) / (green + nir + 1e-6)
    return np.clip(ndwi, -1.0, 1.0)


def compute_ndvi_change(baseline_red: np.ndarray, baseline_nir: np.ndarray, current_red: np.ndarray, current_nir: np.ndarray) -> np.ndarray:
    baseline_ndvi = compute_ndvi(baseline_red, baseline_nir)
    current_ndvi = compute_ndvi(current_red, current_nir)
    return current_ndvi - baseline_ndvi


def save_ndvi_figure(ndvi: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    plt.colorbar(label="NDVI")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_ndvi_change_figure(ndvi_change: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(ndvi_change, cmap="RdBu", vmin=-0.5, vmax=0.5)
    plt.colorbar(label="NDVI Change")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_rgbn_image(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[-1] < 4:
        raise ValueError(f"Expected a 4-band image at {image_path}")
    image = image.astype(np.float32) / 255.0
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    nir = image[..., 3]
    return red, green, blue, nir
