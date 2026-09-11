# Prior-Depth-Anything Evaluation

This directory follows the reference evaluation structure: one CLI, dataset
adapters, shared inference/I/O/metrics, output management, and evaluator
backends. It supports all reference benchmarks (HAMMER, ClearPose, DREDS,
iBims, KITTI) and the project-specific TRansPose adapter. The model adapter calls the native
`prior_depth_anything.PriorDepthAnything.infer_one_sample` interface and stores
metric depth in meters.

The former `evaluation/dataset.py`, `infer.py`, `eval.py`, `evaluation/utils/`,
and `evaluation_ibims/` entry points were removed. Existing runs must be
recreated with the unified CLI because the run layout and prediction names are
versioned and there is no compatibility forwarding layer.

## Environment

```bash
uv sync --extra evaluation --group dev
```

Run every command with `uv run`. The Mac setup is suitable for syntax, imports,
dataset decoding, scoring, and CLI checks. Full inference requires a Linux CUDA
server and the `torch-cluster` wheel selected by `uv sync --extra cuda`.

## Commands

All datasets share one entry point:

```bash
uv run --extra evaluation python -m evaluation <dataset> [options]
```

### HAMMER

```bash
uv run --extra evaluation python -m evaluation hammer \
  --model-path ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-path ckpts/depth_anything_v2_vitl.pth \
  --manifest data/HAMMER/test.jsonl --camera d435
```

`--camera` accepts `d435`, `l515`, or `tof`. Raw and GT PNG values are divided
by 1000 and filtered by each manifest row's `depth-range`.

### ClearPose

```bash
uv run --extra evaluation python -m evaluation clearpose \
  --model-path ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-path ckpts/depth_anything_v2_vitl.pth \
  --manifest data/clearpose/test.jsonl
```

Sequence rows use `rgb`, `rgb-suffix`, `raw_depth-suffix`, `depth-suffix`, and
`depth-range`. Frames are matched by their shared stem.

### DREDS

```bash
uv run --extra evaluation python -m evaluation dreds \
  --model-path ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-path ckpts/depth_anything_v2_vitl.pth \
  --known-manifest data/DREDS/test_std_catknown.jsonl \
  --novel-manifest data/DREDS/test_std_catnovel.jsonl
```

Use `--variants catknown` or `--variants catnovel` to select one subset. EXR
raw and GT values are already meters. Prediction maps are nearest-resized to
GT only for DREDS, matching the reference protocol.

### iBims

```bash
uv run --extra evaluation python -m evaluation ibims \
  --model-path ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-path ckpts/depth_anything_v2_vitl.pth \
  --ibims-root data/ibims1 --levels easy medium hard extreme
```

The root must contain `ibims1_core_mat/`,
`evaluation_scripts/evaluate_ibims.py`, and the per-level JSONL manifests under
`ibims1_synthetic_raw_depth/manifests/`. Predictions are staged as official MAT
files before the evaluator is invoked through the original project-specific
`uv run --project <repository> --no-sync` launcher (`UV_BIN` overrides `uv`).
`--evaluation-seed` (default 0) seeds the official evaluator independently of the
model sampler; the evaluator script hashes are recorded with the result.

### KITTI Depth Completion

```bash
uv run --extra evaluation python -m evaluation.prepare_kitti_jsonl \
  --dataset-root data/KITTI --output data/KITTI/val_selection_cropped.jsonl
uv run --extra evaluation python -m evaluation kitti \
  --model-path ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-path ckpts/depth_anything_v2_vitl.pth \
  --manifest data/KITTI/val_selection_cropped.jsonl --raw-max-depth 80
```

The helper pairs the standard `val_selection_cropped` RGB, Velodyne, GT and
intrinsics files. Depth PNGs are divided by 256; raw input defaults to an 80 m
upper limit, while GT has no upper cutoff. KITTI uses the reference AS-Depth float32 scorer, including finite zero/negative
predictions in the valid GT mask when supplied. The Prior-Depth-Anything adapter
retains its existing prediction normalization (nonfinite/nonpositive -> NaN and
linear restoration to the original RGB grid). This adapter behavior is recorded separately
from the benchmark scorer in `run.json`.

KITTI previews preserve the project's RGB/raw/prediction panel plus color depth
and point clouds. Supply intrinsics in each manifest row or `--intrinsics-path`,
or use `--no-save-visualizations`.

### TRansPose

TRansPose is the current-project adapter at `evaluation/datasets/transpose.py`.
It uses L515 raw depth and GT PNG values divided by 1000, with a default valid
range of 0.1–6.0 m.

```bash
uv run --extra evaluation python -m evaluation transpose \
  --model-path ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-path ckpts/depth_anything_v2_vitl.pth \
  --manifest data/TRansPose/sequences/dc_testset.jsonl
```

## Common options

- `--stage all|infer|evaluate` selects the pipeline stage.
- `--model-path` and `--mde-path` accept local checkpoints; omit either to use the native Hugging Face loader.
- `--run-dir` selects a run directory and is required for evaluate-only runs.
- `--device auto|cuda|mps|cpu` selects the torch device. CUDA is required for real model inference.
- `--frozen-model-size` and `--conditioned-model-size` select the native model backbones.
- `--model-version`, `--coarse-only`, `--pattern`, `--double-global`, `--prior-cover`, and `--down-fill-mode` map directly to native inference options.
- `--seed` records the random seed used by the sparse sampler; the default is `0`.
- `--evaluation-seed` selects the shared iBims scoring seed independently; default `0`.
- `--max-samples N` limits each subset independently for smoke testing.
- `--save-visualizations` and `--no-save-visualizations` control previews.
- `--cleanup-predictions` removes canonical NPY files only after evaluation succeeds.

Convenience wrappers in `evaluation/scripts/` forward to the same `uv run`
entry point.

## Output layout

When `--run-dir` is omitted, runs are written under
`outputs/evaluation/<dataset>/<model_stem>_<YYYYMMDD_HHMMSS>/`:

```text
run.json
predictions/<subset>/<relative_sample_path>.npy
visualizations/<subset>/<relative_sample_path>_vis.jpg
metrics/per_sample.csv
metrics/summary.csv
metrics/summary.json
official/<subset>/predictions/*.mat       # iBims
official/<subset>/workspace/            # iBims
official/<subset>/evaluator.log         # iBims
```

Predictions are float32 metric depth; nonfinite/nonpositive values are stored as
`NaN`, retaining this model adapter's existing output handling. Previews keep
the original three-panel layout and do not read GT, including inference-only
runs. Ground truth is read by the evaluator.
HAMMER, ClearPose, DREDS, and TRansPose report MAE, RMSE, absolute relative
error, and delta accuracy at 1.05, 1.10, and 1.25. Metrics exclude non-finite,
non-positive predictions and invalid GT pixels, then average per-sample scores
within each subset and overall. iBims retains the names emitted by its official
evaluator.


The shared `core/protocol.py` records benchmark protocol and ordered sample
identity in `run.json`. Evaluate-only validates the same selection and scoring
protocol, preserves the original inference configuration, and rejects cleaned or
incomplete predictions. Full manifest hashes are recorded as provenance; edits
that only change visualization or model-specific fields do not invalidate the
benchmark identity. Checkpoints, model settings and sample metadata remain
recorded separately. Image contents are not hashed, so use the same dataset
files for all models. See [ALIGNMENT.md](ALIGNMENT.md) for the behavioral contract
and the allowed project adaptations; matching source files is not required.


Compare completed runs from any of the aligned repositories (benchmark
identity must match; model preprocessing and postprocessing remain visible):

```bash
uv run --extra evaluation python -m evaluation.compare_runs \
  /path/to/model_a/run /path/to/model_b/run --output outputs/comparison.csv
```


The native sampler receives uint8 HWC RGB and float32 metric prior depth without
forcing their spatial sizes to match. It supports low-resolution priors with
`--pattern` omitted; sampling and fill behavior remain native. The adapter moves
native output tensors from the model device to CPU float32 before NumPy export,
then restores the original RGB grid, matching native model output resolution
and preserving evaluation against RGB-aligned GT even with a low-resolution prior. The sampler and tensor-output contract
are covered by a CPU regression test without loading checkpoint weights.
