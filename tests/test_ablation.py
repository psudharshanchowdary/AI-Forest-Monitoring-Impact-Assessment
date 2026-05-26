from pathlib import Path

import numpy as np

from forest_monitor.experiments.ablation import AblationVariant, default_ablation_variants
from forest_monitor.models.evaluation import compute_f1_per_class, label_map_to_detection_prediction



def test_ablation_variants_include_required_cases():
    names = {variant.name for variant in default_ablation_variants()}
    assert {"full_model", "no_ndvi", "no_nir", "ndvi_only", "no_environment"}.issubset(names)



def test_label_map_to_detection_prediction_builds_instances():
    label_map = np.full((32, 32), -1, dtype=np.int16)
    label_map[4:16, 4:16] = 0
    label_map[18:28, 18:28] = 1
    score_stack = np.zeros((32, 32, 5), dtype=np.float32)
    score_stack[..., 0] = 0.2
    score_stack[..., 1] = 0.2
    score_stack[4:16, 4:16, 0] = 0.9
    score_stack[18:28, 18:28, 1] = 0.8
    prediction = label_map_to_detection_prediction(label_map, score_stack=score_stack, min_component_pixels=20)
    assert prediction["labels"].shape[0] == 2
    assert prediction["scores"].shape[0] == 2



def test_compute_f1_per_class_returns_named_scores():
    confusion = np.zeros((6, 6), dtype=np.int64)
    confusion[0, 0] = 10
    confusion[1, 1] = 8
    confusion[0, 1] = 2
    confusion[1, 0] = 1
    scores = compute_f1_per_class(confusion)
    assert "forest" in scores
    assert "field" in scores
    assert 0.0 <= scores["forest"] <= 1.0
