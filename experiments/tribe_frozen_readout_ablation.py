"""Subject-specific readout over frozen public TRIBE encoder features.

The public release checkpoint exposes an average-subject prediction head. This
ablation keeps the multimodal TRIBE encoder frozen and fits only a regularized
linear readout from the model's low-rank latent state to local Schaefer-1000
parcel responses. It tests subject adaptation without changing stimulus
representations or using hidden Friends S7/OOD labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange

from experiments.eval_pretrained_tribe_algonauts import (
    DEFAULT_CACHE,
    DEFAULT_DATA,
    load_stimulus_events,
    vertexwise_pearson,
)
from tribev2.demo_utils import TribeModel


@dataclass(frozen=True)
class RunSpec:
    label: str
    subject: str
    movie: str
    chunk: str
    baseline_npz: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_run_spec(text: str) -> RunSpec:
    """Parse label=subject:movie:chunk:path."""
    if "=" not in text:
        raise ValueError(f"Run spec must be label=subject:movie:chunk:path, got {text!r}")
    label, rest = text.split("=", 1)
    parts = rest.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Run spec must be label=subject:movie:chunk:path, got {text!r}")
    subject, movie, chunk, baseline = parts
    return RunSpec(
        label=label,
        subject=subject,
        movie=movie,
        chunk=chunk,
        baseline_npz=Path(baseline),
    )


def _run_key(spec: RunSpec, features: list[str]) -> str:
    return f"{spec.subject}_{spec.movie}{spec.chunk}_{'_'.join(features)}"


def run_identity(spec: RunSpec) -> tuple[str, str, str]:
    return (spec.subject, spec.movie, spec.chunk)


def episode_identity(spec: RunSpec) -> tuple[str, str, str]:
    match = re.match(r"^(e\d+)", spec.chunk)
    episode = match.group(1) if match else spec.chunk
    return (spec.subject, spec.movie, episode)


def validate_disjoint_runs(train_specs: list[RunSpec], val_specs: list[RunSpec]) -> None:
    train_ids = {run_identity(spec) for spec in train_specs}
    val_ids = {run_identity(spec) for spec in val_specs}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Train/validation run overlap: {sorted(overlap)}")
    train_episode_ids = {episode_identity(spec) for spec in train_specs}
    val_episode_ids = {episode_identity(spec) for spec in val_specs}
    episode_overlap = train_episode_ids & val_episode_ids
    if episode_overlap:
        raise ValueError(
            f"Train/validation episode overlap: {sorted(episode_overlap)}"
        )


def _scalar_npz_value(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        raise ValueError(f"Cached feature file is missing metadata field {key!r}")
    value = data[key]
    if value.shape != ():
        raise ValueError(f"Cached feature metadata field {key!r} is not scalar")
    return str(value.item())


def _json_sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".json")


def load_baseline_metadata(spec: RunSpec) -> dict[str, Any]:
    path = _json_sidecar_path(spec.baseline_npz)
    if not path.exists():
        raise FileNotFoundError(f"{spec.label}: missing baseline metadata sidecar {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_baseline_metadata(
    spec: RunSpec,
    metadata: dict[str, Any],
    payload: dict[str, np.ndarray],
    *,
    expected_checkpoint: str,
    expected_features: list[str],
    expected_metric_space: str,
    expected_target_offset_seconds: float,
) -> None:
    expected = {
        "checkpoint": expected_checkpoint,
        "features": list(expected_features),
        "metric_space": expected_metric_space,
        "subject": spec.subject,
        "movie": spec.movie,
        "chunks": [spec.chunk],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{spec.label}: baseline metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    observed_offset = float(metadata.get("target_offset_seconds"))
    if not math.isclose(
        observed_offset,
        float(expected_target_offset_seconds),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{spec.label}: target offset mismatch "
            f"{observed_offset} != {expected_target_offset_seconds}"
        )
    arrays_path = metadata.get("arrays_path")
    if arrays_path and Path(arrays_path).resolve() != spec.baseline_npz.resolve():
        raise ValueError(
            f"{spec.label}: metadata arrays_path {arrays_path!r} does not match "
            f"{str(spec.baseline_npz)!r}"
        )
    if int(metadata.get("n_prediction_samples")) != int(payload["preds"].shape[0]):
        raise ValueError(f"{spec.label}: baseline row count metadata mismatch")
    if list(metadata.get("metric_prediction_shape", [])) != list(payload["preds"].shape):
        raise ValueError(f"{spec.label}: baseline prediction shape metadata mismatch")
    if list(metadata.get("target_shape", [])) != list(payload["target"].shape):
        raise ValueError(f"{spec.label}: baseline target shape metadata mismatch")


def validate_feature_alignment(
    spec: RunSpec,
    feature_payload: dict[str, Any],
    baseline_payload: dict[str, np.ndarray],
    *,
    target_offset_seconds: float,
    atol: float = 1e-4,
) -> None:
    shifted = feature_payload["segment_starts"] + float(target_offset_seconds)
    sample_times = baseline_payload["sample_times"]
    if shifted.shape != sample_times.shape:
        raise ValueError(
            f"{spec.label}: feature/sample time shape mismatch "
            f"{shifted.shape} vs {sample_times.shape}"
        )
    if not np.allclose(shifted, sample_times, atol=atol, rtol=0.0):
        max_error = float(np.max(np.abs(shifted - sample_times)))
        raise ValueError(
            f"{spec.label}: feature/sample time mismatch; max error {max_error:.6g}"
        )


def low_rank_features_for_batch(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    """Return pooled pre-predictor latent features, shape B,C,T_out."""
    x = model.aggregate_features(batch)
    subject_id = batch.data.get("subject_id", None)
    if hasattr(model, "temporal_smoothing"):
        x = model.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
    if not model.config.linear_baseline:
        x = model.transformer_forward(x, subject_id)
    if hasattr(model, "temporal_delay_bank"):
        x = model.temporal_delay_bank(x)
    x = x.transpose(1, 2)
    if hasattr(model, "low_rank_head"):
        x = model.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
    return model.pooler(x)


def extract_low_rank_features(
    model_wrapper: TribeModel,
    events: Any,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract kept-segment latent features and segment start times."""
    if model_wrapper._model is None:
        raise RuntimeError("TribeModel has no loaded torch model")
    model = model_wrapper._model
    loader = model_wrapper.data.get_loaders(events=events, split_to_build="all")["all"]

    features, starts = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            batch_segments = []
            for segment in batch.segments:
                for t in np.arange(0, segment.duration - 1e-2, model_wrapper.data.TR):
                    batch_segments.append(segment.copy(offset=t, duration=model_wrapper.data.TR))
            if model_wrapper.remove_empty_segments:
                keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
            else:
                keep = np.ones(len(batch_segments), dtype=bool)
            if not keep.any():
                continue
            latent = low_rank_features_for_batch(model, batch).detach().cpu().numpy()
            latent = rearrange(latent, "b d t -> (b t) d")[keep]
            kept_segments = [s for i, s in enumerate(batch_segments) if keep[i]]
            features.append(latent.astype(np.float32))
            starts.extend(float(getattr(segment, "start", i)) for i, segment in enumerate(kept_segments))
    if not features:
        raise ValueError("No non-empty latent segments were extracted")
    return np.concatenate(features, axis=0), np.asarray(starts, dtype=np.float32)


def load_or_extract_features(
    spec: RunSpec,
    *,
    data_path: Path,
    checkpoint: str,
    model_wrapper: TribeModel,
    features: list[str],
    feature_dir: Path,
    force: bool,
    target_offset_seconds: float,
) -> dict[str, Any]:
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"tribe_lowrank_{_run_key(spec, features)}.npz"
    if path.exists() and not force:
        data = np.load(path)
        if _scalar_npz_value(data, "checkpoint") != checkpoint:
            raise ValueError(f"{spec.label}: cached feature checkpoint mismatch")
        if _scalar_npz_value(data, "subject") != spec.subject:
            raise ValueError(f"{spec.label}: cached feature subject mismatch")
        if _scalar_npz_value(data, "movie") != spec.movie:
            raise ValueError(f"{spec.label}: cached feature movie mismatch")
        if _scalar_npz_value(data, "chunk") != spec.chunk:
            raise ValueError(f"{spec.label}: cached feature chunk mismatch")
        cached_features = json.loads(_scalar_npz_value(data, "features_json"))
        if cached_features != list(features):
            raise ValueError(f"{spec.label}: cached feature modality mismatch")
        cached_offset = float(_scalar_npz_value(data, "target_offset_seconds"))
        if not math.isclose(
            cached_offset,
            float(target_offset_seconds),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{spec.label}: cached feature target offset mismatch")
        return {
            "path": path,
            "features": data["features"].astype(np.float32),
            "segment_starts": data["segment_starts"].astype(np.float32),
            "used_cache": True,
        }

    events = load_stimulus_events(
        data_path,
        spec.subject,
        spec.movie,
        [spec.chunk],
        features,
    )
    feats, starts = extract_low_rank_features(model_wrapper, events)
    np.savez_compressed(
        path,
        features=feats,
        segment_starts=starts,
        checkpoint=np.asarray(checkpoint),
        subject=np.asarray(spec.subject),
        movie=np.asarray(spec.movie),
        chunk=np.asarray(spec.chunk),
        features_json=np.asarray(json.dumps(list(features))),
        target_offset_seconds=np.asarray(float(target_offset_seconds)),
        grid=np.asarray("tribe_lowrank_segment_starts"),
        cache_schema=np.asarray("v2"),
    )
    return {
        "path": path,
        "features": feats,
        "segment_starts": starts,
        "used_cache": False,
    }


def load_baseline_run(spec: RunSpec) -> dict[str, np.ndarray]:
    data = np.load(spec.baseline_npz)
    return {
        "preds": data["preds"].astype(np.float32),
        "target": data["target"].astype(np.float32),
        "sample_times": data["sample_times"].astype(np.float32),
        "finite_rows": data["finite_rows"].astype(bool),
    }


def finite_unit_mask(runs: list[dict[str, np.ndarray]]) -> np.ndarray:
    if not runs:
        raise ValueError("No runs provided")
    n_units = runs[0]["target"].shape[1]
    mask = np.ones(n_units, dtype=bool)
    for run in runs:
        rows = run["finite_rows"]
        mask &= np.isfinite(run["target"][rows]).all(axis=0)
        mask &= np.isfinite(run["preds"][rows]).all(axis=0)
    return mask


def prepare_xy(
    specs: list[RunSpec],
    feature_payloads: dict[str, dict[str, Any]],
    baseline_payloads: dict[str, dict[str, np.ndarray]],
    unit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for spec in specs:
        feats = feature_payloads[spec.label]["features"]
        base = baseline_payloads[spec.label]
        rows = base["finite_rows"]
        if feats.shape[0] != base["target"].shape[0]:
            raise ValueError(
                f"{spec.label}: feature/target row mismatch "
                f"{feats.shape[0]} vs {base['target'].shape[0]}"
            )
        xs.append(feats[rows])
        ys.append(base["target"][rows][:, unit_mask])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_ridge_readout(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y row mismatch: {x.shape[0]} vs {y.shape[0]}")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    weights = np.linalg.solve(gram, x_centered.T @ y_centered)
    return {
        "weights": weights.astype(np.float32),
        "x_mean": x_mean.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
    }


def predict_ridge(x: np.ndarray, fitted: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (x.astype(np.float32) - fitted["x_mean"]) @ fitted["weights"]
        + fitted["y_mean"]
    ).astype(np.float32)


def score_baseline(
    specs: list[RunSpec],
    baseline_payloads: dict[str, dict[str, np.ndarray]],
    unit_mask: np.ndarray,
) -> dict[str, Any]:
    preds, targets = [], []
    for spec in specs:
        run = baseline_payloads[spec.label]
        rows = run["finite_rows"]
        preds.append(run["preds"][rows][:, unit_mask])
        targets.append(run["target"][rows][:, unit_mask])
    return vertexwise_pearson(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))


def select_alpha(
    train_specs: list[RunSpec],
    feature_payloads: dict[str, dict[str, Any]],
    baseline_payloads: dict[str, dict[str, np.ndarray]],
    unit_mask: np.ndarray,
    alphas: list[float],
) -> dict[str, Any]:
    if len(train_specs) < 2:
        return {"selected_alpha": float(alphas[0]), "inner_scores": []}
    inner_scores = []
    for alpha in alphas:
        fold_scores = []
        for holdout in train_specs:
            inner_train = [spec for spec in train_specs if spec != holdout]
            x_train, y_train = prepare_xy(
                inner_train, feature_payloads, baseline_payloads, unit_mask
            )
            x_val, y_val = prepare_xy(
                [holdout], feature_payloads, baseline_payloads, unit_mask
            )
            fitted = fit_ridge_readout(x_train, y_train, alpha=alpha)
            pred = predict_ridge(x_val, fitted)
            fold_scores.append(vertexwise_pearson(pred, y_val)["mean"])
        inner_scores.append(
            {
                "alpha": float(alpha),
                "mean_score": float(np.nanmean(fold_scores)),
                "fold_scores": [float(v) for v in fold_scores],
            }
        )
    best = max(inner_scores, key=lambda row: row["mean_score"])
    return {"selected_alpha": float(best["alpha"]), "inner_scores": inner_scores}


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    train_specs = [parse_run_spec(text) for text in args.train_run]
    val_specs = [parse_run_spec(text) for text in args.val_run]
    validate_disjoint_runs(train_specs, val_specs)
    all_specs = train_specs + val_specs
    features = list(args.features)

    model = TribeModel.from_pretrained(
        args.checkpoint,
        cache_folder=args.cache_folder,
        cluster=None,
        device=args.device,
        config_update={
            "data.features_to_use": features,
            "data.num_workers": 0,
            "data.batch_size": 1,
            "enable_progress_bar": False,
        },
    )

    feature_payloads = {
        spec.label: load_or_extract_features(
            spec,
            data_path=args.data_path,
            checkpoint=args.checkpoint,
            model_wrapper=model,
            features=features,
            feature_dir=args.feature_dir,
            force=args.force_features,
            target_offset_seconds=args.target_offset_seconds,
        )
        for spec in all_specs
    }
    baseline_payloads = {spec.label: load_baseline_run(spec) for spec in all_specs}
    baseline_metadata = {spec.label: load_baseline_metadata(spec) for spec in all_specs}
    for spec in all_specs:
        validate_baseline_metadata(
            spec,
            baseline_metadata[spec.label],
            baseline_payloads[spec.label],
            expected_checkpoint=args.checkpoint,
            expected_features=features,
            expected_metric_space="parcel",
            expected_target_offset_seconds=args.target_offset_seconds,
        )
    for spec in all_specs:
        validate_feature_alignment(
            spec,
            feature_payloads[spec.label],
            baseline_payloads[spec.label],
            target_offset_seconds=args.target_offset_seconds,
        )
    unit_mask = finite_unit_mask([baseline_payloads[spec.label] for spec in all_specs])
    alpha_result = select_alpha(
        train_specs,
        feature_payloads,
        baseline_payloads,
        unit_mask,
        [float(alpha) for alpha in args.alpha],
    )
    selected_alpha = float(alpha_result["selected_alpha"])
    x_train, y_train = prepare_xy(train_specs, feature_payloads, baseline_payloads, unit_mask)
    x_val, y_val = prepare_xy(val_specs, feature_payloads, baseline_payloads, unit_mask)
    fitted = fit_ridge_readout(x_train, y_train, alpha=selected_alpha)
    pred_val = predict_ridge(x_val, fitted)
    val_readout = vertexwise_pearson(pred_val, y_val)
    val_baseline = score_baseline(val_specs, baseline_payloads, unit_mask)
    train_readout = vertexwise_pearson(predict_ridge(x_train, fitted), y_train)
    train_baseline = score_baseline(train_specs, baseline_payloads, unit_mask)
    delta_val = (
        float(val_readout["mean"] - val_baseline["mean"])
        if val_readout["mean"] is not None and val_baseline["mean"] is not None
        else None
    )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_frozen_readout_ablation",
        "checkpoint": args.checkpoint,
        "features": features,
        "train_runs": [_jsonable(spec.__dict__) for spec in train_specs],
        "val_runs": [_jsonable(spec.__dict__) for spec in val_specs],
        "n_train_rows": int(x_train.shape[0]),
        "n_val_rows": int(x_val.shape[0]),
        "latent_dim": int(x_train.shape[1]),
        "n_scored_units_common_with_baseline": int(unit_mask.sum()),
        "alpha_grid": [float(alpha) for alpha in args.alpha],
        "target_offset_seconds": float(args.target_offset_seconds),
        "split_policy": "episode-disjoint available-label local proxy",
        "metric_space": "approx_projected_schaefer1000_parcel_1hz_interpolated_targets",
        "selected_alpha": selected_alpha,
        "inner_alpha_selection": alpha_result,
        "train_readout": train_readout,
        "train_baseline": train_baseline,
        "val_readout": val_readout,
        "val_baseline": val_baseline,
        "delta_val_readout_vs_baseline": delta_val,
        "feature_artifacts": {
            label: {
                "path": payload["path"],
                "shape": list(payload["features"].shape),
                "used_cache": payload["used_cache"],
            }
            for label, payload in feature_payloads.items()
        },
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy only. The readout is trained on local "
            "Algonauts labels and scored on held-out available chunks. Hidden "
            "Friends S7/OOD labels were not used."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--features", nargs="+", default=["text", "audio", "video"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-dir", type=Path, default=Path("cache/ablation_results/readout_features"))
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--target-offset-seconds", type=float, default=5.0)
    parser.add_argument("--train-run", nargs="+", required=True)
    parser.add_argument("--val-run", nargs="+", required=True)
    parser.add_argument("--alpha", nargs="+", type=float, default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    parser.add_argument("--out", type=Path, default=Path("cache/ablation_results/tribe_frozen_readout_ablation.json"))
    args = parser.parse_args()

    result = run_ablation(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
