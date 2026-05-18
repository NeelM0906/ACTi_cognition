"""Run a small Algonauts text-only loss ablation.

This is a benchmark smoke harness, not the final leaderboard run. It uses the
Algonauts2025 parcellated HDF5 data, restricts features to text, and compares
training objectives on the same queried subset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from exca import ConfDict


DEFAULT_DATA = Path("/home/ripper/data/tribe_benchmarks")
DEFAULT_RUNS = Path("/home/ripper/data/tribe_runs")
DEFAULT_QUERY = (
    "subject == 'Algonauts2025/sub-01' and task == 'friends' and "
    "movie == 's01' and chunk in ['e01a', 'e01b']"
)
SUPPORTED_FEATURES = ("text", "audio", "video", "image")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _build_transforms(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build only the event transforms required by the selected modalities."""
    transforms: dict[str, dict[str, Any]] = {}
    features = set(args.features_to_use)
    if "audio" in features:
        transforms["extractaudio"] = {"name": "ExtractAudioFromVideo"}

    transforms.update(
        {
            "addsentence": {
                "name": "AddSentenceToWords",
                "max_unmatched_ratio": 0.05,
            },
            "addcontext": {
                "name": "AddContextToWords",
                "sentence_only": False,
                "max_context_len": args.max_context_len,
                "split_field": "",
            },
            "removemissing": {"name": "RemoveMissing"},
        }
    )
    if "audio" in features:
        transforms["chunksounds"] = {
            "name": "ChunkEvents",
            "event_type_to_chunk": "Audio",
            "max_duration": 60,
            "min_duration": 30,
        }
    if "video" in features or "image" in features:
        transforms["chunkvideos"] = {
            "name": "ChunkEvents",
            "event_type_to_chunk": "Video",
            "max_duration": 60,
            "min_duration": 30,
        }

    if args.split_mode == "chunk":
        split_transform = {
            "name": "AssignSplitByValue",
            "field": "chunk",
            "val_values": list(args.val_chunks),
        }
    elif args.split_mode == "episode":
        split_transform = {
            "name": "AssignSplitByEpisode",
            "chunk_field": "chunk",
            "movie_field": "movie",
            "val_episodes": list(args.val_episodes),
        }
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")
    transforms["split"] = split_transform
    return transforms


def _cache_version_for(args: argparse.Namespace, modality: str) -> str:
    specific = getattr(args, f"{modality}_cache_version", None)
    if specific:
        return specific
    if modality == "text":
        return args.text_cache_version
    if args.feature_cache_version:
        return args.feature_cache_version
    return args.text_cache_version


def _configure_feature_extractors(config: ConfDict, args: argparse.Namespace) -> None:
    for modality in args.features_to_use:
        extractor = config["data"].get(f"{modality}_feature")
        if extractor is None:
            raise ValueError(f"No extractor configured for modality {modality!r}")
        extractor["infra"]["cluster"] = None
        extractor["infra"]["version"] = _cache_version_for(args, modality)

    if args.text_model_name:
        config["data"]["text_feature"]["model_name"] = args.text_model_name
    if args.text_layers_json:
        config["data"]["text_feature"]["layers"] = json.loads(args.text_layers_json)
    if args.text_cache_n_layers is not None:
        config["data"]["text_feature"]["cache_n_layers"] = args.text_cache_n_layers
    if args.text_batch_size is not None:
        config["data"]["text_feature"]["batch_size"] = args.text_batch_size
    if args.audio_layers_json:
        config["data"]["audio_feature"]["layers"] = json.loads(args.audio_layers_json)
    if args.video_model_name:
        config["data"]["video_feature"]["image"]["model_name"] = args.video_model_name
    if args.video_layers_json:
        config["data"]["video_feature"]["image"]["layers"] = json.loads(
            args.video_layers_json
        )
    if args.video_batch_size is not None:
        config["data"]["video_feature"]["image"]["batch_size"] = args.video_batch_size


def build_config(args: argparse.Namespace, loss_name: str, seed: int) -> ConfDict:
    os.environ.setdefault("DATAPATH", str(args.data_path))
    os.environ.setdefault("SAVEPATH", str(args.run_root))

    from tribev2.grids.configs import mini_config

    config = ConfDict(copy.deepcopy(mini_config))
    folder = args.run_root / args.run_tag / loss_name / f"seed_{seed}"
    ablation_transforms = _build_transforms(args)
    loss_config: dict[str, Any]
    if loss_name == "mse":
        loss_config = {"name": "MSELoss", "kwargs": {"reduction": "none"}}
    elif loss_name == "pearson_mse":
        loss_config = {
            "name": "PearsonMSELoss",
            "pearson_weight": args.pearson_weight,
            "mse_weight": args.mse_weight,
            "dim": 0,
            "reduction": "none",
        }
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    config.update(
        {
            "infra.cluster": None,
            "infra.folder": str(folder),
            "infra.mode": "force",
            "infra.workdir": None,
            "wandb_config": None,
            "save_checkpoints": False,
            "n_epochs": args.epochs,
            "limit_train_batches": args.limit_train_batches,
            "seed": seed,
            "log_every_n_steps": 1,
            "data.num_workers": 0,
            "data.batch_size": args.batch_size,
            "data.study.names": "Algonauts2025",
            "data.study.path": str(args.data_path),
            "data.study.query": args.query,
            "data.study.infra_timelines.version": args.timeline_cache_version,
            "data.features_to_use": list(args.features_to_use),
            "data.neuro.projection": None,
            "data.neuro.infra.cluster": None,
            "data.neuro.infra.folder": None,
            "loss": loss_config,
            "monitor": "val/pearson",
        }
    )
    _configure_feature_extractors(config, args)
    if args.data_layer_aggregation:
        config["data"]["layer_aggregation"] = (
            None
            if args.data_layer_aggregation.lower() == "none"
            else args.data_layer_aggregation
        )
    del config["data"]["study"]["transforms"]
    config["data"]["study"]["transforms"] = ablation_transforms
    del config["loss"]
    config["loss"] = loss_config
    if args.tiny_model:
        config.update(
            {
                "brain_model_config.hidden": 256,
                "brain_model_config.encoder.depth": 2,
                "brain_model_config.low_rank_head": 128,
            }
        )
    if args.temporal_delay_bank:
        config.update(
            {
                "brain_model_config.temporal_delay_bank": {
                    "name": "TemporalDelayBank",
                    "n_lags": args.delay_lags,
                    "lag_stride": args.delay_stride,
                    "per_channel": args.delay_per_channel,
                    "init_identity": True,
                }
            }
        )
    if args.output_temporal_delay_bank:
        config.update(
            {
                "brain_model_config.output_temporal_delay_bank": {
                    "name": "TemporalDelayBank",
                    "n_lags": args.output_delay_lags,
                    "lag_stride": args.output_delay_stride,
                    "per_channel": args.output_delay_per_channel,
                    "init_identity": True,
                }
            }
        )
    return config


def run_one(args: argparse.Namespace, loss_name: str, seed: int) -> dict[str, Any]:
    from tribev2.main import TribeExperiment

    config = build_config(args, loss_name, seed)
    task = TribeExperiment(**config)
    task.infra.clear_job()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    task.run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else None
    )
    metrics = {}
    if task._trainer is not None:
        metrics = {
            key: value
            for key, value in task._trainer.callback_metrics.items()
            if not key.endswith("/step")
        }
    return {
        "loss": loss_name,
        "seed": seed,
        "seconds": seconds,
        "peak_vram_mb": peak_vram_mb,
        "folder": config["infra.folder"],
        "metrics": _jsonable(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--val-chunks", nargs="+", default=["e01b"])
    parser.add_argument("--val-episodes", nargs="+", default=["e01"])
    parser.add_argument("--split-mode", choices=["chunk", "episode"], default="chunk")
    parser.add_argument("--seeds", nargs="+", type=int, default=[33])
    parser.add_argument("--run-tag", default="text_loss_ablation")
    parser.add_argument("--timeline-cache-version", default="text_loss_ablation_v1")
    parser.add_argument("--text-cache-version", default="text_loss_ablation_v2")
    parser.add_argument("--feature-cache-version")
    parser.add_argument("--audio-cache-version")
    parser.add_argument("--video-cache-version")
    parser.add_argument(
        "--features-to-use",
        nargs="+",
        choices=SUPPORTED_FEATURES,
        default=["text"],
    )
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--pearson-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.05)
    parser.add_argument("--text-model-name")
    parser.add_argument(
        "--text-layers-json",
        help="JSON float or list for data.text_feature.layers.",
    )
    parser.add_argument("--text-cache-n-layers", type=int)
    parser.add_argument("--text-batch-size", type=int)
    parser.add_argument("--audio-layers-json")
    parser.add_argument("--video-model-name")
    parser.add_argument("--video-layers-json")
    parser.add_argument("--video-batch-size", type=int)
    parser.add_argument("--data-layer-aggregation")
    parser.add_argument("--loss", choices=["mse", "pearson_mse", "both"], default="both")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--temporal-delay-bank", action="store_true")
    parser.add_argument("--delay-lags", type=int, default=6)
    parser.add_argument("--delay-stride", type=int, default=2)
    parser.add_argument(
        "--delay-per-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-temporal-delay-bank", action="store_true")
    parser.add_argument("--output-delay-lags", type=int, default=6)
    parser.add_argument("--output-delay-stride", type=int, default=2)
    parser.add_argument(
        "--output-delay-per-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/algonauts_text_loss_ablation.json"),
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)

    losses = ["mse", "pearson_mse"] if args.loss == "both" else [args.loss]
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "query": args.query,
        "split_mode": args.split_mode,
        "val_chunks": args.val_chunks,
        "val_episodes": args.val_episodes,
        "seeds": args.seeds,
        "run_tag": args.run_tag,
        "timeline_cache_version": args.timeline_cache_version,
        "text_cache_version": args.text_cache_version,
        "feature_cache_version": args.feature_cache_version,
        "audio_cache_version": args.audio_cache_version,
        "video_cache_version": args.video_cache_version,
        "features_to_use": args.features_to_use,
        "max_context_len": args.max_context_len,
        "matmul_precision": args.matmul_precision,
        "temporal_delay_bank": args.temporal_delay_bank,
        "delay_lags": args.delay_lags,
        "delay_stride": args.delay_stride,
        "delay_per_channel": args.delay_per_channel,
        "output_temporal_delay_bank": args.output_temporal_delay_bank,
        "output_delay_lags": args.output_delay_lags,
        "output_delay_stride": args.output_delay_stride,
        "output_delay_per_channel": args.output_delay_per_channel,
        "pearson_weight": args.pearson_weight,
        "mse_weight": args.mse_weight,
        "text_model_name": args.text_model_name,
        "text_layers": json.loads(args.text_layers_json)
        if args.text_layers_json
        else None,
        "text_cache_n_layers": args.text_cache_n_layers,
        "text_batch_size": args.text_batch_size,
        "audio_layers": json.loads(args.audio_layers_json)
        if args.audio_layers_json
        else None,
        "video_model_name": args.video_model_name,
        "video_layers": json.loads(args.video_layers_json)
        if args.video_layers_json
        else None,
        "video_batch_size": args.video_batch_size,
        "data_layer_aggregation": (
            None
            if args.data_layer_aggregation
            and args.data_layer_aggregation.lower() == "none"
            else args.data_layer_aggregation
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "limit_train_batches": args.limit_train_batches,
        "tiny_model": args.tiny_model,
        "runs": [
            run_one(args, loss_name, seed)
            for seed in args.seeds
            for loss_name in losses
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    print(args.out)
    print(json.dumps(_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
