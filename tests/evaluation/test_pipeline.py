import cv2
import numpy as np

from evaluation.core.pipeline import run_pipeline
from evaluation.core.types import EvaluationSample, RunConfig
from evaluation.datasets.base import DatasetCollection


class FakeModel:
    def infer_one_sample(self, image, prior, **_kwargs):
        assert image.shape[:2] == prior.shape
        return prior.copy()


def test_pipeline_smoke_scores_fixed_prediction(tmp_path, monkeypatch):
    rgb = tmp_path / "rgb.png"
    raw = tmp_path / "raw.png"
    gt = tmp_path / "gt.png"
    assert cv2.imwrite(str(rgb), np.full((4, 5, 3), 127, dtype=np.uint8))
    assert cv2.imwrite(str(raw), np.full((4, 5), 1000, dtype=np.uint16))
    assert cv2.imwrite(str(gt), np.full((4, 5), 1000, dtype=np.uint16))
    sample = EvaluationSample("scene/frame", "d435", rgb, raw, gt, 1000.0, 0.1, 5.0)
    collection = DatasetCollection("hammer", [sample])
    config = RunConfig(
        "hammer", "all", tmp_path / "run", "model.pt", device="cpu", save_visualizations=False
    )
    monkeypatch.setattr("evaluation.core.inference.load_model", lambda *_args: FakeModel())
    layout = run_pipeline(collection, config)
    assert layout.prediction_path(sample).is_file()
    assert layout.metrics_dir.joinpath("summary.json").is_file()


def test_infer_previews_keep_three_panels_and_do_not_require_gt(tmp_path, monkeypatch):
    rgb = tmp_path / "rgb.png"
    raw = tmp_path / "raw.png"
    assert cv2.imwrite(str(rgb), np.full((4, 5, 3), 127, dtype=np.uint8))
    assert cv2.imwrite(str(raw), np.full((4, 5), 1000, dtype=np.uint16))
    sample = EvaluationSample(
        "scene/frame", "d435", rgb, raw, tmp_path / "unavailable_gt.png", 1000, 0.1, 5
    )
    config = RunConfig("hammer", "infer", tmp_path / "run", "model.pt", device="cpu")
    monkeypatch.setattr("evaluation.core.inference.load_model", lambda *_: FakeModel())
    layout = run_pipeline(DatasetCollection("hammer", [sample]), config)
    assert cv2.imread(str(layout.visualization_path(sample))).shape == (4, 15, 3)



def test_native_sampler_accepts_low_resolution_prior_and_tensor_output(tmp_path, monkeypatch):
    import torch

    from prior_depth_anything.sparse_sampler import SparseSampler

    class NativeSamplerModel:
        def infer_one_sample(self, image, prior, **kwargs):
            assert image.shape == (8, 10, 3) and image.dtype == np.uint8
            assert prior.shape == (2, 3) and prior.dtype == np.float32
            assert kwargs["pattern"] is None
            data = SparseSampler(device="cpu")(image=image, prior=prior, K=5)
            assert data["sparse_depth"].shape == (1, 1, 8, 10)
            assert data["sparse_mask"].sum() == 6
            # Real inference returns a device Tensor. bfloat16 also requires
            # conversion before NumPy can consume it, even in this CPU test.
            return torch.full((8, 10), 1.25, dtype=torch.bfloat16)

    rgb, raw, gt = (tmp_path / filename for filename in ("rgb.png", "raw.png", "gt.png"))
    assert cv2.imwrite(str(rgb), np.full((8, 10, 3), 127, dtype=np.uint8))
    assert cv2.imwrite(str(raw), np.full((2, 3), 1250, dtype=np.uint16))
    assert cv2.imwrite(str(gt), np.full((8, 10), 1250, dtype=np.uint16))
    sample = EvaluationSample("scene/frame", "d435", rgb, raw, gt, 1000, 0.1, 5)
    config = RunConfig(
        "hammer", "all", tmp_path / "run", "model.pt",
        device="cpu", save_visualizations=False,
    )
    monkeypatch.setattr("evaluation.core.inference.load_model", lambda *_: NativeSamplerModel())
    layout = run_pipeline(DatasetCollection("hammer", [sample]), config)
    prediction = np.load(layout.prediction_path(sample))
    assert prediction.shape == (8, 10)
    np.testing.assert_array_equal(prediction, np.full((8, 10), 1.25, dtype=np.float32))
    import json

    summary = json.loads((layout.metrics_dir / "summary.json").read_text())
    assert summary["overall"]["rmse"] == 0
