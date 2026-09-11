from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class EvaluationSample:
    """One RGB/raw-depth evaluation sample."""

    sample_id: str
    subset: str
    rgb_path: Path
    raw_depth_path: Path
    gt_depth_path: Optional[Path]
    depth_scale: float
    min_depth: float
    max_depth: float
    allow_evaluation_resize: bool = False
    expected_shape: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    raw_max_depth: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    stage: str
    run_dir: Path
    model_path: Optional[str]
    mde_path: Optional[str] = None
    device: str = "auto"
    model_version: str = "1.1"
    frozen_model_size: str = "vitl"
    conditioned_model_size: str = "vitb"
    coarse_only: bool = False
    pattern: Optional[str] = None
    double_global: bool = False
    prior_cover: bool = False
    down_fill_mode: str = "linear"
    seed: int = 0
    batch_size: int = 1
    num_workers: int = 0
    save_visualizations: bool = True
    cleanup_predictions: bool = False
    max_samples: Optional[int] = None
    visualization_min_depth: float = 0.1
    visualization_max_depth: float = 5.0
    intrinsics_path: Optional[Path] = None
    pointcloud_rot_x_deg: float = 25.0
    pointcloud_rot_y_deg: float = 15.0
    pointcloud_knn_k: int = 16
    pointcloud_knn_std_ratio: float = 2.0
    disable_pointcloud_knn_filter: bool = False
    evaluation_seed: int = 0


@dataclass(frozen=True)
class LoadedSample:
    sample: EvaluationSample
    rgb: Any
    raw_depth: Any
    gt_depth: Any = None
