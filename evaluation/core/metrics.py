from __future__ import annotations

from typing import Dict

import numpy as np

METRIC_NAMES = (
    "mae",
    "rmse",
    "abs_rel",
    "delta_1_05",
    "delta_1_10",
    "delta_1_25",
)


def compute_depth_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    # AS-Depth's evaluator computes the delta ratios with float32 Torch
    # tensors. Compute only the ratios at that precision for strict threshold
    # behavior, while retaining float64 aggregates for stable error reporting.
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Metric shape mismatch: prediction={pred.shape}, target={gt.shape}")

    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0.0) & (gt > 0.0)
    if not np.any(valid):
        return {name: float("nan") for name in METRIC_NAMES}

    pred_valid = pred[valid]
    gt_valid = gt[valid]
    difference = pred_valid - gt_valid
    ratio = np.maximum(
        np.asarray(pred_valid, dtype=np.float32) / np.asarray(gt_valid, dtype=np.float32),
        np.asarray(gt_valid, dtype=np.float32) / np.asarray(pred_valid, dtype=np.float32),
    )

    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "abs_rel": float(np.mean(np.abs(difference) / gt_valid)),
        "delta_1_05": float(np.mean(ratio < 1.05)),
        "delta_1_10": float(np.mean(ratio < 1.10)),
        "delta_1_25": float(np.mean(ratio < 1.25)),
    }
