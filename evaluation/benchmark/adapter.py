"""Prior-Depth-Anything 完整粗细两阶段 forward；sampler 在计时前运行。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .support import PreparedForward

PROJECT = "priorda"
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_FIELDS = ("checkpoint", "mde_checkpoint")
DEFAULT_MODEL = {
    "checkpoint": None,
    "mde_checkpoint": None,
    "version": "1.1",
    "frozen_model_size": "vitl",
    "conditioned_model_size": "vitb",
}


def validate_options(options: dict[str, Any]) -> None:
    if (options["version"], options["frozen_model_size"], options["conditioned_model_size"]) != (
        "1.1",
        "vitl",
        "vitb",
    ):
        raise ValueError("v1 benchmark selects Prior v1.1 with frozen vitl + conditioned vitb")


def load(options: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    from prior_depth_anything import PriorDepthAnything

    model = PriorDepthAnything(
        device=device,
        version=options["version"],
        mde_path=options["mde_checkpoint"],
        ckpt_path=options["checkpoint"],
        frozen_model_size=options["frozen_model_size"],
        conditioned_model_size=options["conditioned_model_size"],
        coarse_only=False,
    ).eval()
    return model, {
        "class": "PriorDepthAnything",
        "stages": ["frozen_mde", "conditioned_mde"],
        "coarse_only": False,
    }


def prepare(model: Any, rgb: Any, depth: Any, config: dict[str, Any]) -> PreparedForward:
    model.args.double_global = False
    data = model.sampler(
        image=rgb,
        prior=depth,
        geometric=None,
        pattern=None,
        K=model.args.K,
        prior_cover=False,
        down_fill_mode="linear",
    )
    if not bool(data["sparse_mask"].all()):
        raise ValueError("100% valid synthetic depth must not be sparsified for Prior")
    kwargs = {
        "images": data["rgb"],
        "sparse_depths": data["sparse_depth"],
        "sparse_masks": data["sparse_mask"],
        "cover_masks": data["cover_mask"],
        "prior_depths": data["prior_depth"],
        "geometric_depths": None,
        "pattern": None,
    }
    return PreparedForward(
        model,
        kwargs=kwargs,
        metadata={
            "entrypoint": "PriorDepthAnything.forward",
            "coarse_only": False,
            "double_global": False,
            "knn_k": model.args.K,
            "knn_query_pixels": 0,
            "internal_input_size": 518,
        },
    )
