"""Faithful local TRIBE retrain harness.

This harness is the baseline entry point for the v2 research queue. It keeps
the TRIBE experiment machinery from ``tribev2.main`` and the defaults from
``tribev2.grids.defaults``, then applies only local-run constraints: explicit
Algonauts subject/episode selection, episode-disjoint validation, local caches,
and a shape/finite smoke gate over parcellated Schaefer1000 outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from exca import ConfDict


DEFAULT_DATA = Path("/home/ripper/data/tribe_benchmarks")
DEFAULT_RUNS = Path("/home/ripper/data/tribe_runs")
DEFAULT_FEATURES = ("text", "audio", "video")
STUDY_NAME_MAP = {
    "algonauts2025": "Algonauts2025",
    "algonauts2025bold": "Algonauts2025Bold",
    "lahner2024bold": "Lahner2024Bold",
    "lebel2023bold": "Lebel2023Bold",
    "wen2017": "Wen2017",
}
EPISODE_RE = re.compile(r"^s(?P<season>\d{2})e(?P<episode>\d{2})$")
SEASON_RE = re.compile(r"^s(?P<season>\d{2})$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
        ).strip()
    except Exception:
        return True
    return bool(status)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _expand_numbered_range(
    token: str,
    *,
    regex: re.Pattern[str],
    formatter,
    label: str,
) -> list[str]:
    if ".." not in token:
        if not regex.match(token):
            raise ValueError(f"Invalid {label}: {token!r}")
        return [token]
    start_text, end_text = token.split("..", 1)
    start_match = regex.match(start_text)
    end_match = regex.match(end_text)
    if not start_match or not end_match:
        raise ValueError(f"Invalid {label} range: {token!r}")
    start = {key: int(value) for key, value in start_match.groupdict().items()}
    end = {key: int(value) for key, value in end_match.groupdict().items()}
    shared_keys = [key for key in start.keys() if key != label]
    for key in shared_keys:
        if start[key] != end[key]:
            raise ValueError(f"Cross-{key} {label} ranges are not supported: {token!r}")
    start_idx = start[label]
    end_idx = end[label]
    if end_idx < start_idx:
        raise ValueError(f"Descending {label} ranges are not supported: {token!r}")
    return [formatter(start | {label: idx}) for idx in range(start_idx, end_idx + 1)]


def expand_episodes(values: list[str]) -> list[str]:
    episodes: list[str] = []
    for value in values:
        episodes.extend(
            _expand_numbered_range(
                value,
                regex=EPISODE_RE,
                formatter=lambda item: f"s{item['season']:02d}e{item['episode']:02d}",
                label="episode",
            )
        )
    return episodes


def expand_seasons(values: list[str]) -> list[str]:
    seasons: list[str] = []
    for value in values:
        seasons.extend(
            _expand_numbered_range(
                value,
                regex=SEASON_RE,
                formatter=lambda item: f"s{item['season']:02d}",
                label="season",
            )
        )
    return seasons


def default_val_episodes(episodes: list[str], explicit: list[str] | None) -> list[str]:
    if explicit:
        return expand_episodes(explicit)
    if len(episodes) < 2:
        raise ValueError(
            "At least two episodes are required when --val-episodes is omitted."
        )
    return [episodes[-1]]


def chunks_for_episode(episode: str) -> list[str]:
    match = EPISODE_RE.match(episode)
    if not match:
        raise ValueError(f"Invalid episode id: {episode!r}")
    episode_num = int(match.group("episode"))
    return [f"e{episode_num:02d}{suffix}" for suffix in "abcd"]


def build_algonauts_query(subjects: list[str], episodes: list[str]) -> str:
    if not subjects:
        raise ValueError("At least one subject is required")
    if not episodes:
        raise ValueError("At least one episode is required")

    subject_values = [f"Algonauts2025/{subject}" for subject in subjects]
    seasons = sorted({episode[:3] for episode in episodes})
    chunks = sorted({chunk for episode in episodes for chunk in chunks_for_episode(episode)})
    clauses = [
        f"subject in {subject_values!r}",
        "task == 'friends'",
        f"movie in {seasons!r}",
        f"chunk in {chunks!r}",
    ]
    return " and ".join(clauses)


def build_episode_split_transform(val_episodes: list[str]) -> dict[str, Any]:
    return {
        "name": "AssignSplitByEpisode",
        "chunk_field": "chunk",
        "movie_field": "movie",
        "val_episodes": list(val_episodes),
    }


def _set_local_infra(config: ConfDict, args: argparse.Namespace) -> None:
    config.update(
        {
            "infra.cluster": None,
            "infra.mode": "force",
            "infra.workdir": None,
            "wandb_config": None,
            "data.num_workers": args.num_workers,
            "data.study.infra_timelines.cluster": None,
            "data.study.infra_timelines.version": args.timeline_cache_version,
            "data.neuro.infra.cluster": None,
            "data.neuro.infra.folder": None,
            "data.neuro.infra.version": args.timeline_cache_version,
            "log_every_n_steps": 1,
            "enable_progress_bar": args.enable_progress_bar,
        }
    )
    for modality in ("text", "audio", "video", "image"):
        extractor = config["data"].get(f"{modality}_feature")
        if extractor is not None:
            extractor["infra"]["cluster"] = None
            extractor["infra"]["version"] = args.feature_cache_version


def _apply_feature_overrides(config: ConfDict, args: argparse.Namespace) -> None:
    if args.video_features == "vjepa2":
        config["data"]["video_feature"]["image"]["model_name"] = (
            "facebook/vjepa2-vitg-fpc64-256"
        )
        config["data"]["video_feature"]["image"]["layers"] = [0.75, 1.0]
    elif args.video_features != "default":
        raise ValueError(f"Unsupported --video-features: {args.video_features}")

    if args.audio_features != "default":
        raise ValueError(f"Unsupported --audio-features: {args.audio_features}")
    if args.features_source != "default":
        raise ValueError(f"Unsupported --features-source: {args.features_source}")
    if args.text_layer_aggregation != "default":
        raise ValueError(
            f"Unsupported --text-layer-aggregation: {args.text_layer_aggregation}"
        )
    if args.text_backbone:
        config["data"]["text_feature"]["model_name"] = args.text_backbone


def build_config(args: argparse.Namespace, seed: int) -> tuple[ConfDict, dict[str, Any]]:
    os.environ.setdefault("DATAPATH", str(args.data_path))
    os.environ.setdefault("SAVEPATH", str(args.run_root))

    from tribev2.grids.defaults import default_config

    episodes = expand_episodes(args.episodes or [])
    seasons = expand_seasons(args.seasons or [])
    if seasons and episodes:
        raise ValueError("Use --episodes or --seasons, not both")
    if seasons:
        raise NotImplementedError("--seasons support is reserved for H002 scale-up")
    if not episodes:
        raise ValueError("--episodes is required for the current retrain harness")
    val_episodes = default_val_episodes(episodes, args.val_episodes)

    studies = [STUDY_NAME_MAP.get(name.lower(), name) for name in args.studies]
    if studies != ["Algonauts2025"]:
        raise NotImplementedError(
            "The first harness rung supports Algonauts2025 parcellated retrains; "
            "multi-study scale-up will extend this entry point."
        )

    config = ConfDict(copy.deepcopy(default_config))
    run_folder = args.run_root / args.run_tag / f"seed_{seed}"
    query = args.query or build_algonauts_query(args.subjects, episodes)

    config.update(
        {
            "infra.folder": str(run_folder),
            "data.study.names": "Algonauts2025",
            "data.study.path": str(args.data_path),
            "data.study.query": query,
            "data.features_to_use": list(args.features_to_use),
            "data.neuro.projection": None,
            "data.batch_size": args.batch_size,
            "data.duration_trs": args.duration_trs,
            "seed": seed,
            "n_epochs": args.epochs,
            "limit_train_batches": args.limit_train_batches,
            "save_checkpoints": args.save_checkpoints,
        }
    )
    transforms = copy.deepcopy(config["data"]["study"]["transforms"])
    del transforms["split"]
    transforms["split"] = build_episode_split_transform(val_episodes)
    del config["data"]["study"]["transforms"]
    config["data"]["study"]["transforms"] = transforms
    _set_local_infra(config, args)
    _apply_feature_overrides(config, args)

    if args.tiny_model:
        config.update(
            {
                "brain_model_config.hidden": 256,
                "brain_model_config.encoder.depth": 2,
                "brain_model_config.low_rank_head": 128,
            }
        )

    metadata = {
        "subjects": args.subjects,
        "episodes": episodes,
        "val_episodes": val_episodes,
        "studies": studies,
        "query": query,
        "run_folder": run_folder,
    }
    return config, metadata


def config_hash(config: ConfDict) -> str:
    payload = json.dumps(_jsonable(config.to_dict()), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _move_batch_to_device(batch, device: torch.device):
    if hasattr(batch, "to"):
        return batch.to(device)
    for key, value in batch.data.items():
        if isinstance(value, torch.Tensor):
            batch.data[key] = value.to(device)
    return batch


def smoke_prediction_summary(task) -> dict[str, Any]:
    loaders = task.data.get_loaders(split_to_build="val")
    if "val" not in loaders:
        raise RuntimeError("No validation loader was built for smoke shape check")
    batch = next(iter(loaders["val"]))
    model = task._model
    if model is None:
        raise RuntimeError("Task model is not initialized after run")
    device = next(model.parameters()).device
    batch = _move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        y_pred = model(batch).detach().cpu()
    if y_pred.ndim != 3:
        raise ValueError(f"Expected B,D,T prediction tensor, got {tuple(y_pred.shape)}")
    rows = rearrange(y_pred, "b d t -> (b t) d").numpy()
    finite = bool(np.isfinite(rows).all())
    return {
        "batch_prediction_shape_bdt": list(y_pred.shape),
        "prediction_shape": list(rows.shape),
        "n_TRs": int(rows.shape[0]),
        "n_outputs": int(rows.shape[1]),
        "all_finite": finite,
        "passes_shape_finite_gate": finite and rows.shape[1] == 1000,
    }


def run_one(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    from tribev2.main import TribeExperiment

    config, metadata = build_config(args, seed)
    task = TribeExperiment(**config)
    task.infra.clear_job()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    task.run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - start
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
    shape_summary = smoke_prediction_summary(task)
    folder = Path(config["infra.folder"])
    checkpoints = sorted(str(path) for path in folder.glob("*.ckpt"))
    return {
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mb": peak_vram_mb,
        "folder": str(folder),
        "checkpoints": checkpoints,
        "metrics": _jsonable(metrics),
        "shape_summary": shape_summary,
        "config_hash": config_hash(config),
        "metadata": _jsonable(metadata),
    }


def primary_value_from_runs(runs: list[dict[str, Any]]) -> Any:
    pearsons = []
    for run in runs:
        metrics = run.get("metrics", {})
        for key in ("test/pearson", "val/pearson"):
            if key in metrics:
                try:
                    pearsons.append(float(metrics[key]))
                    break
                except (TypeError, ValueError):
                    pass
    if pearsons:
        return float(np.mean(pearsons))
    if all(run["shape_summary"]["passes_shape_finite_gate"] for run in runs):
        shapes = [tuple(run["shape_summary"]["prediction_shape"]) for run in runs]
        return ";".join(f"({shape[0]},{shape[1]});finite" for shape in shapes)
    return "shape_finite_failed"


def write_manifest(args: argparse.Namespace, result_path: Path, results: dict[str, Any]) -> Path:
    manifest_path = args.manifest_out
    if manifest_path is None:
        stem = result_path.stem
        manifest_path = Path("research/manifests") / f"{stem}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": results["created_at"],
        "script": "experiments/tribe_faithful_retrain.py",
        "result_json": str(result_path),
        "git_sha": results["git_sha"],
        "git_dirty": results["git_dirty"],
        "host": socket.gethostname(),
        "data_path": str(args.data_path),
        "run_root": str(args.run_root),
        "cache_policy": "local cached feature/timeline extractors; no hidden labels",
        "runs": [
            {
                "seed": run["seed"],
                "folder": run["folder"],
                "checkpoints": run["checkpoints"],
                "shape_summary": run["shape_summary"],
                "runtime_seconds": run["runtime_seconds"],
                "peak_vram_mb": run["peak_vram_mb"],
            }
            for run in results["runs"]
        ],
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--run-tag", default="tribe_faithful_retrain")
    parser.add_argument("--studies", nargs="+", default=["algonauts2025"])
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument("--seasons", nargs="+")
    parser.add_argument("--val-episodes", nargs="+")
    parser.add_argument("--val-policy", choices=["episode-heldout"], default="episode-heldout")
    parser.add_argument("--query")
    parser.add_argument("--seeds", nargs="+", type=int, default=[33])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--duration-trs", type=int, default=100)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--features-to-use",
        nargs="+",
        choices=DEFAULT_FEATURES,
        default=list(DEFAULT_FEATURES),
    )
    parser.add_argument("--timeline-cache-version", default="h001_faithful_retrain_v1")
    parser.add_argument("--feature-cache-version", default="h001_faithful_retrain_v1")
    parser.add_argument("--video-features", default="default")
    parser.add_argument("--audio-features", default="default")
    parser.add_argument("--features-source", default="default")
    parser.add_argument("--text-layer-aggregation", default="default")
    parser.add_argument("--text-backbone")
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument("--manifest-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    timestamp = _utc_timestamp()
    out = args.out or Path(
        f"cache/ablation_results/tribe_faithful_retrain_{timestamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    runs = [run_one(args, seed) for seed in args.seeds]
    total_runtime = time.perf_counter() - start
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "command_args": _jsonable(vars(args)),
        "matmul_precision": args.matmul_precision,
        "primary_metric": "heldout_val_pearson_mean"
        if any("test/pearson" in run.get("metrics", {}) for run in runs)
        else "shape_finite",
        "primary_value": primary_value_from_runs(runs),
        "gate": "shape (n_TRs, 1000) finite",
        "gate_passed": all(
            run["shape_summary"]["passes_shape_finite_gate"] for run in runs
        ),
        "runtime_seconds": total_runtime,
        "runs": runs,
    }
    out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    manifest_path = write_manifest(args, out, results)
    results["manifest_path"] = str(manifest_path)
    out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
