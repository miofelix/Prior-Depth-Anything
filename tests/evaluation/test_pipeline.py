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
