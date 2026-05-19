from argparse import Namespace
from pathlib import Path

import numpy as np

from experiments.tribe_faithful_retrain import (
    build_algonauts_query,
    build_config,
    default_val_episodes,
    expand_episodes,
    primary_value_from_runs,
)


def _args(tmp_path: Path, **overrides):
    args = {
        "data_path": tmp_path / "data",
        "run_root": tmp_path / "runs",
        "run_tag": "test_h001",
        "studies": ["algonauts2025"],
        "subjects": ["sub-01"],
        "episodes": ["s01e01", "s01e02"],
        "seasons": None,
        "val_episodes": None,
        "val_policy": "episode-heldout",
        "query": None,
        "seeds": [33],
        "epochs": 1,
        "batch_size": 1,
        "duration_trs": 16,
        "limit_train_batches": 1,
        "num_workers": 0,
        "features_to_use": ["text", "audio", "video"],
        "timeline_cache_version": "timeline_test",
        "feature_cache_version": "feature_test",
        "video_features": "default",
        "audio_features": "default",
        "features_source": "default",
        "text_layer_aggregation": "default",
        "text_backbone": None,
        "matmul_precision": "high",
        "tiny_model": True,
        "save_checkpoints": True,
        "enable_progress_bar": False,
        "out": None,
        "manifest_out": None,
    }
    args.update(overrides)
    return Namespace(**args)


def test_expand_episodes_accepts_compact_ranges():
    assert expand_episodes(["s01e01..s01e03", "s01e05"]) == [
        "s01e01",
        "s01e02",
        "s01e03",
        "s01e05",
    ]


def test_default_val_episode_uses_last_episode():
    assert default_val_episodes(["s01e01", "s01e02"], None) == ["s01e02"]
    assert default_val_episodes(["s01e01", "s01e02"], ["s01e01"]) == ["s01e01"]


def test_build_algonauts_query_selects_subject_episode_halves():
    query = build_algonauts_query(["sub-01", "sub-02"], ["s01e01", "s01e02"])

    assert "Algonauts2025/sub-01" in query
    assert "Algonauts2025/sub-02" in query
    assert "movie in ['s01']" in query
    assert "e01a" in query
    assert "e02d" in query


def test_build_config_uses_defaults_with_episode_split_and_parcel_targets(tmp_path):
    config, metadata = build_config(_args(tmp_path), seed=33)

    assert config["data"]["study"]["names"] == "Algonauts2025"
    assert config["data"]["study"]["query"] == metadata["query"]
    assert config["data"]["study"]["transforms"]["split"]["name"] == (
        "AssignSplitByEpisode"
    )
    assert config["data"]["study"]["transforms"]["split"]["val_episodes"] == [
        "s01e02"
    ]
    assert "val_ratio" not in config["data"]["study"]["transforms"]["split"]
    assert config["data"]["neuro"]["projection"] is None
    assert config["data"]["neuro"]["infra"]["folder"] is None
    assert config["data"]["features_to_use"] == ["text", "audio", "video"]
    assert config["data"]["text_feature"]["infra"]["cluster"] is None
    assert config["data"]["text_feature"]["infra"]["version"] == "feature_test"
    assert config["brain_model_config"]["hidden"] == 256
    assert config["brain_model_config"]["encoder"]["depth"] == 2


def test_primary_value_reports_shape_gate_when_no_metric():
    runs = [
        {
            "shape_summary": {
                "prediction_shape": [16, 1000],
                "passes_shape_finite_gate": True,
            },
            "metrics": {},
        }
    ]

    assert primary_value_from_runs(runs) == "(16,1000);finite"


def test_primary_value_prefers_mean_pearson_metric():
    runs = [
        {"metrics": {"test/pearson": np.float32(0.1)}, "shape_summary": {}},
        {"metrics": {"test/pearson": np.float32(0.3)}, "shape_summary": {}},
    ]

    assert np.isclose(primary_value_from_runs(runs), 0.2)
