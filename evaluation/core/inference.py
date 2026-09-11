from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.core.io import (
    normalize_prediction,
    read_raw_depth,
    read_rgb,
)
from evaluation.core.output import RunLayout, save_prediction
from evaluation.core.types import EvaluationSample, LoadedSample, RunConfig
from evaluation.core.visualization import (
    save_kitti_pointcloud_visualization,
    save_kitti_prediction_visualization,
    save_visualization,
)
from evaluation.datasets.base import DatasetCollection

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):
        return iterable


class InferenceInputDataset(Dataset):
    def __init__(self, samples: Sequence[EvaluationSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LoadedSample:
        sample = self.samples[index]
        raw_depth = read_raw_depth(
            sample.raw_depth_path,
            sample.depth_scale,
            sample.min_depth,
            sample.raw_max_depth if sample.raw_max_depth is not None else sample.max_depth,
        )
        return LoadedSample(sample=sample, rgb=read_rgb(sample.rgb_path), raw_depth=raw_depth)


def collate_loaded_samples(items: List[LoadedSample]) -> List[LoadedSample]:
    return items


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def load_model(config: RunConfig, device: torch.device):
    if device.type != "cuda":
        raise RuntimeError(
            "Prior-Depth-Anything inference requires CUDA and torch-cluster; "
            "CPU/MPS are available only for evaluation and basic checks"
        )
    from prior_depth_anything import PriorDepthAnything

    return PriorDepthAnything(
        device=str(device),
        version=config.model_version,
        mde_path=config.mde_path,
        ckpt_path=config.model_path,
        frozen_model_size=config.frozen_model_size,
        conditioned_model_size=config.conditioned_model_size,
        coarse_only=config.coarse_only,
    ).eval()


@torch.inference_mode()
def run_inference(
    collection: DatasetCollection, config: RunConfig, layout: RunLayout
) -> Dict[str, object]:
    if config.batch_size < 1 or config.num_workers < 0:
        raise ValueError("invalid batch or worker count")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = select_device(config.device)
    model = load_model(config, device)
    if collection.name == "kitti" and config.save_visualizations:
        for sample in collection.samples:
            if not (sample.metadata.get("intrinsics_path") or config.intrinsics_path):
                raise ValueError(
                    "KITTI visualization requires an intrinsics path in every manifest row "
                    "or --intrinsics-path"
                )
    loader = DataLoader(
        InferenceInputDataset(collection.samples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_loaded_samples,
    )
    written = 0
    for batch in tqdm(loader, desc=f"{collection.name} inference"):
        for item in batch:
            sample = item.sample
            if sample.expected_shape is not None and item.rgb.shape[:2] != sample.expected_shape:
                raise ValueError(
                    f"Unexpected RGB shape for {sample.sample_id}: "
                    f"got {item.rgb.shape[:2]}, expected {sample.expected_shape}"
                )
            prediction = model.infer_one_sample(
                image=item.rgb,
                prior=item.raw_depth,
                geometric=None,
                pattern=config.pattern,
                double_global=config.double_global,
                prior_cover=config.prior_cover,
                visualize=False,
                down_fill_mode=config.down_fill_mode,
            )
            # The native model returns a Tensor on its inference device.
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach().float().cpu().numpy()
            prediction = normalize_prediction(prediction, item.rgb.shape[:2])
            save_prediction(layout.prediction_path(sample), prediction)
            if config.save_visualizations:
                save_visualization(
                    layout.visualization_path(sample),
                    item.rgb,
                    item.raw_depth,
                    prediction,
                    None,
                    config.visualization_min_depth,
                    config.visualization_max_depth,
                )
                if collection.name == "kitti":
                    save_kitti_prediction_visualization(
                        layout.kitti_prediction_visualization_path(sample), prediction,
                        config.visualization_min_depth, config.visualization_max_depth,
                    )
                    intrinsics_value = sample.metadata.get("intrinsics_path")
                    intrinsics_path = (
                        Path(intrinsics_value) if intrinsics_value else config.intrinsics_path
                    )
                    assert intrinsics_path is not None
                    save_kitti_pointcloud_visualization(
                        layout.kitti_pointcloud_visualization_path(sample), item.rgb, prediction,
                        intrinsics_path, config.pointcloud_rot_x_deg, config.pointcloud_rot_y_deg,
                        config.pointcloud_knn_k, config.pointcloud_knn_std_ratio,
                        config.disable_pointcloud_knn_filter,
                    )
            written += 1
    return {
        "num_predictions": written,
        "device": str(device),
        "seed": config.seed,
        "preprocessing": {
            "rgb": "uint8 HWC RGB; native SparseSampler owns spatial adaptation",
            "prompt": "float32 metric depth at its original resolution",
            "low_resolution_prompt": "native SparseSampler downscale path with pattern=None",
            "gt_used_for_inference": False,
        },
        "prediction_postprocessing": {
            "normalization": "normalize_prediction",
            "invalid_values": "nonfinite or nonpositive -> NaN",
            "resize": "linear prediction with nearest validity mask to original RGB grid",
            "depth_unit": "meter",
            "scale_shift_fit": False,
            "clipping": False,
        },
        "model_class": "prior_depth_anything.PriorDepthAnything",
        "gt_used_for_inference": False,
    }
