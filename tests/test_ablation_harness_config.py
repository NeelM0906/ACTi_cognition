from argparse import Namespace
from pathlib import Path

from experiments.algonauts_text_loss_ablation import build_config


def _args(tmp_path: Path, **overrides):
    args = {
        "data_path": tmp_path / "data",
        "run_root": tmp_path / "runs",
        "run_tag": "test_run",
        "query": (
            "subject == 'Algonauts2025/sub-01' and task == 'friends' and "
            "movie == 's01' and chunk in ['e01a', 'e01b']"
        ),
        "epochs": 1,
        "batch_size": 1,
        "limit_train_batches": 1,
        "val_chunks": ["e01b"],
        "val_episodes": ["s01e01"],
        "split_mode": "chunk",
        "timeline_cache_version": "timeline_test",
        "text_cache_version": "text_test",
        "feature_cache_version": None,
        "audio_cache_version": None,
        "video_cache_version": None,
        "features_to_use": ["text"],
        "max_context_len": 1024,
        "pearson_weight": 1.0,
        "mse_weight": 0.05,
        "text_model_name": None,
        "text_layers_json": None,
        "text_cache_n_layers": None,
        "text_batch_size": None,
        "audio_layers_json": None,
        "video_model_name": None,
        "video_layers_json": None,
        "video_batch_size": None,
        "data_layer_aggregation": None,
        "tiny_model": True,
        "temporal_delay_bank": False,
        "delay_lags": 6,
        "delay_stride": 2,
        "delay_per_channel": True,
        "output_temporal_delay_bank": False,
        "output_delay_lags": 6,
        "output_delay_stride": 2,
        "output_delay_per_channel": True,
    }
    args.update(overrides)
    return Namespace(**args)


def test_text_only_harness_avoids_audio_transforms(tmp_path):
    config = build_config(_args(tmp_path), "mse", 33)

    transforms = config["data"]["study"]["transforms"]
    assert config["data"]["features_to_use"] == ["text"]
    assert "extractaudio" not in transforms
    assert "chunksounds" not in transforms
    assert "addcontext" in transforms
    assert "split" in transforms
    assert config["data"]["text_feature"]["infra"]["cluster"] is None
    assert config["data"]["text_feature"]["infra"]["version"] == "text_test"


def test_text_audio_harness_adds_audio_transforms_and_cache(tmp_path):
    config = build_config(
        _args(
            tmp_path,
            features_to_use=["text", "audio"],
            audio_cache_version="audio_test",
            audio_layers_json="[0.75, 1.0]",
        ),
        "mse",
        33,
    )

    transforms = config["data"]["study"]["transforms"]
    assert config["data"]["features_to_use"] == ["text", "audio"]
    assert "extractaudio" in transforms
    assert "chunksounds" in transforms
    assert transforms["chunksounds"]["event_type_to_chunk"] == "Audio"
    assert config["data"]["audio_feature"]["infra"]["cluster"] is None
    assert config["data"]["audio_feature"]["infra"]["version"] == "audio_test"
    assert config["data"]["audio_feature"]["layers"] == [0.75, 1.0]
