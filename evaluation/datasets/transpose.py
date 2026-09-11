from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import (
    DatasetCollection,
    limit_per_subset,
    normalize_sample_id,
    parse_depth_range,
    read_jsonl,
    require_keys,
    resolve_path,
    sample_id_from_path,
)

TRANSPOSE_DEFAULT_DEPTH_RANGE = (0.1, 6.0)
def load_transpose(manifest: Path, max_samples: Optional[int] = None) -> DatasetCollection:
    """TRansPose L515 per-frame manifest, with millimeter PNG raw depth and GT."""
    manifest = manifest.expanduser().resolve()
    root = manifest.parent
    samples = []
    for index, row in enumerate(read_jsonl(manifest), start=1):
        context = f"{manifest}:{index}"
        require_keys(row, ("rgb", "l515_depth", "depth"), context)
        min_depth, max_depth = parse_depth_range(
            {"depth-range": TRANSPOSE_DEFAULT_DEPTH_RANGE, **row}, context
        )
        rgb_path = resolve_path(root, row["rgb"])
        sample_id = (
            normalize_sample_id(str(row["seq_name"]))
            if row.get("seq_name")
            else sample_id_from_path(rgb_path, root)
        )
        samples.append(
            EvaluationSample(
                sample_id=sample_id,
                subset="l515",
                rgb_path=rgb_path,
                raw_depth_path=resolve_path(root, row["l515_depth"]),
                gt_depth_path=resolve_path(root, row["depth"]),
                depth_scale=1000.0,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        )
    return DatasetCollection(
        name="transpose",
        samples=limit_per_subset(samples, max_samples),
        metadata={"manifest": str(manifest), "camera": "l515"},
    )
