from pathlib import Path

import pytest

from evaluation.cli import build_parser, resolve_run_dir


def test_cli_exposes_current_dataset_commands():
    parser = build_parser()
    args = parser.parse_args(
        ["transpose", "--manifest", "data/test.jsonl", "--model-path", "model.pt"]
    )
    assert args.dataset == "transpose"
    assert args.model_version == "1.1"
    assert args.seed == 0


def test_evaluate_only_requires_explicit_run_dir():
    parser = build_parser()
    args = parser.parse_args(["hammer", "--manifest", "data/test.jsonl", "--stage", "evaluate"])
    with pytest.raises(SystemExit):
        resolve_run_dir(args, parser)


def test_explicit_run_dir_is_preserved():
    parser = build_parser()
    args = parser.parse_args(["ibims", "--stage", "evaluate", "--run-dir", "outputs/run"])
    assert resolve_run_dir(args, parser) == Path("outputs/run")


def test_run_config_retains_existing_positional_device_slot(tmp_path):
    from evaluation.core.types import RunConfig

    config = RunConfig("hammer", "infer", tmp_path, "model.pt", "mde.pt", "cuda")
    assert config.mde_path == "mde.pt"
    assert config.device == "cuda"
    assert config.evaluation_seed == 0
