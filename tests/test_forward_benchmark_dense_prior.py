"""全有效 prior 的空 KNN 边界；无需真实权重或 CUDA。"""

import pytest
import torch

from prior_depth_anything.depth_completion import DepthCompletion


def completer():
    model = DepthCompletion.__new__(DepthCompletion)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    return model


def test_full_prior_preserves_values_without_calling_knn(monkeypatch):
    model = completer()
    raw = torch.arange(1, 7, dtype=torch.float32).reshape(1, 2, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty query must not enter KNN")

    monkeypatch.setattr(model, "knn_aligns", forbidden)
    output = model.kss_completer(
        raw,
        raw * 2,
        torch.zeros_like(raw, dtype=torch.bool),
        torch.ones_like(raw, dtype=torch.bool),
    )
    torch.testing.assert_close(output, raw)
    output.zero_()
    assert bool((raw > 0).all())


def test_nonempty_query_still_uses_native_knn(monkeypatch):
    model = completer()
    raw = torch.ones(1, 2, 3)
    missing = torch.zeros_like(raw, dtype=torch.bool)
    missing[0, 0, 0] = True

    def called(*args, **kwargs):
        raise RuntimeError("native KNN reached")

    monkeypatch.setattr(model, "knn_aligns", called)
    with pytest.raises(RuntimeError, match="native KNN reached"):
        model.kss_completer(raw, raw, missing, ~missing)
