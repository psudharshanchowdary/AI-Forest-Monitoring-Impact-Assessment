"""Lightweight shared constants for labels and class indices."""

from __future__ import annotations

CLASS_NAMES = ["forest", "field", "lake", "road", "building"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
