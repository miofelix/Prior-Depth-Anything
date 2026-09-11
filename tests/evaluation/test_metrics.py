import math

import numpy as np

from evaluation.core.metrics import compute_depth_metrics


def test_fixed_prediction_metrics_follow_shared_protocol():
    target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    prediction = np.array([[1.0, 1.0], [np.nan, 4.0]], dtype=np.float32)
    metrics = compute_depth_metrics(prediction, target)
    assert math.isclose(metrics["mae"], 1.0 / 3.0)
    assert math.isclose(metrics["abs_rel"], 1.0 / 6.0)
    assert metrics["delta_1_25"] == 2.0 / 3.0


def test_empty_valid_mask_is_nan():
    metrics = compute_depth_metrics(np.array([[np.nan]], dtype=np.float32), np.ones((1, 1)))
    assert all(math.isnan(value) for value in metrics.values())
