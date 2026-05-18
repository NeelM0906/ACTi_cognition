"""Native-TR subject adaptation over public TRIBE predictions/features.

The earlier subject-adaptation ablations scored on TRIBE's 1 Hz segment grid
with fMRI targets interpolated to those segment times. This harness evaluates
the same idea on the native Algonauts fMRI TR grid: public TRIBE predictions
and frozen low-rank features are interpolated to each recorded fMRI sample,
then closed-form ridge readouts are fit using only available training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.eval_pretrained_tribe_algonauts import (
    DEFAULT_DATA,
    load_parcel_data,
    vertexwise_pearson,
)
from experiments.tribe_frozen_readout_ablation import (
    RunSpec,
    _jsonable,
    _scalar_npz_value,
    finite_unit_mask,
    fit_ridge_readout,
    load_baseline_metadata,
    load_baseline_run,
    parse_run_spec,
    predict_ridge,
    prepare_xy,
    score_baseline,
    select_alpha,
    validate_baseline_metadata,
    validate_disjoint_runs,
    validate_feature_alignment,
)
from tribev2.studies.algonauts2025 import Algonauts2025


def resample_to_native_grid(
    values: np.ndarray,
    sample_times: np.ndarray,
    *,
    n_samples: int,
    native_frequency: float,
) -> np.ndarray:
    """Interpolate time-major values to native fMRI sample times.

    Values outside the available predicted time span are edge-filled. This
    mirrors the official submission harness behavior, where a prediction must
    be provided for every requested target sample.
    """
    from scipy.interpolate import interp1d

    if values.ndim != 2:
        raise ValueError(f"Expected 2-D values, got {values.shape}")
    if values.shape[0] != sample_times.shape[0]:
        raise ValueError(
            f"Value/sample-time mismatch: {values.shape[0]} vs {sample_times.shape[0]}"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if values.shape[0] == 0:
        raise ValueError("Cannot resample empty values")

    order = np.argsort(sample_times)
    x = sample_times[order].astype(np.float64)
    y = values[order].astype(np.float32)
    keep = np.concatenate([[True], np.diff(x) > 1e-6])
    x = x[keep]
    y = y[keep]
    if len(x) == 1:
        return np.repeat(y, n_samples, axis=0).astype(np.float32)

    native_times = np.arange(n_samples, dtype=np.float32) / float(native_frequency)
    interp = interp1d(
        x,
        y,
        axis=0,
        bounds_error=False,
        fill_value=(y[0], y[-1]),
        assume_sorted=True,
    )
    return interp(native_times).astype(np.float32)


def _feature_path(spec: RunSpec, features: list[str], feature_dir: Path) -> Path:
    feature_key = "_".join(features)
    return feature_dir / f"tribe_lowrank_{spec.subject}_{spec.movie}{spec.chunk}_{feature_key}.npz"


def load_lowrank_feature_cache(
    spec: RunSpec,
    *,
    checkpoint: str,
    features: list[str],
    feature_dir: Path,
    target_offset_seconds: float,
) -> dict[str, Any]:
    path = _feature_path(spec, features, feature_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{spec.label}: missing low-rank feature cache {path}; "
            "run experiments.tribe_frozen_readout_ablation first"
        )
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
    }


def native_payload_from_baseline(
    spec: RunSpec,
    *,
    data_path: Path,
    baseline_payload: dict[str, np.ndarray],
    native_frequency: float,
) -> dict[str, np.ndarray]:
    target = load_parcel_data(data_path, spec.subject, spec.movie, spec.chunk)
    preds = resample_to_native_grid(
        baseline_payload["preds"],
        baseline_payload["sample_times"],
        n_samples=target.shape[0],
        native_frequency=native_frequency,
    )
    return {
        "preds": preds.astype(np.float32),
        "target": target.astype(np.float32),
        "sample_times": (
            np.arange(target.shape[0], dtype=np.float32) / float(native_frequency)
        ).astype(np.float32),
        "finite_rows": (
            np.isfinite(target).any(axis=1) & np.isfinite(preds).any(axis=1)
        ),
    }


def native_lowrank_features(
    spec: RunSpec,
    feature_payload: dict[str, Any],
    native_payload: dict[str, np.ndarray],
    *,
    native_frequency: float,
    target_offset_seconds: float,
) -> dict[str, np.ndarray]:
    shifted_times = feature_payload["segment_starts"] + float(target_offset_seconds)
    if shifted_times.shape != feature_payload["features"].shape[:1]:
        raise ValueError(f"{spec.label}: low-rank feature/time row mismatch")
    feats = resample_to_native_grid(
        feature_payload["features"],
        shifted_times.astype(np.float32),
        n_samples=native_payload["target"].shape[0],
        native_frequency=native_frequency,
    )
    return {"features": feats.astype(np.float32)}


def fit_and_score(
    train_specs: list[RunSpec],
    val_specs: list[RunSpec],
    feature_payloads: dict[str, dict[str, Any]],
    native_payloads: dict[str, dict[str, np.ndarray]],
    unit_mask: np.ndarray,
    alphas: list[float],
) -> dict[str, Any]:
    alpha_result = select_alpha(
        train_specs,
        feature_payloads,
        native_payloads,
        unit_mask,
        alphas,
    )
    selected_alpha = float(alpha_result["selected_alpha"])
    x_train, y_train = prepare_xy(
        train_specs, feature_payloads, native_payloads, unit_mask
    )
    x_val, y_val = prepare_xy(val_specs, feature_payloads, native_payloads, unit_mask)
    fitted = fit_ridge_readout(x_train, y_train, alpha=selected_alpha)
    train_score = vertexwise_pearson(predict_ridge(x_train, fitted), y_train)
    val_score = vertexwise_pearson(predict_ridge(x_val, fitted), y_val)
    return {
        "selected_alpha": selected_alpha,
        "inner_alpha_selection": alpha_result,
        "train_score": train_score,
        "val_score": val_score,
    }


def run_native_adaptation(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    train_specs = [parse_run_spec(text) for text in args.train_run]
    val_specs = [parse_run_spec(text) for text in args.val_run]
    validate_disjoint_runs(train_specs, val_specs)
    all_specs = train_specs + val_specs
    features = list(args.features)
    alphas = [float(alpha) for alpha in args.alpha]
    native_frequency = float(Algonauts2025._FREQUENCY)

    segment_payloads = {spec.label: load_baseline_run(spec) for spec in all_specs}
    baseline_metadata = {spec.label: load_baseline_metadata(spec) for spec in all_specs}
    for spec in all_specs:
        validate_baseline_metadata(
            spec,
            baseline_metadata[spec.label],
            segment_payloads[spec.label],
            expected_checkpoint=args.checkpoint,
            expected_features=features,
            expected_metric_space="parcel",
            expected_target_offset_seconds=args.target_offset_seconds,
        )

    lowrank_segment_payloads = {
        spec.label: load_lowrank_feature_cache(
            spec,
            checkpoint=args.checkpoint,
            features=features,
            feature_dir=args.feature_dir,
            target_offset_seconds=args.target_offset_seconds,
        )
        for spec in all_specs
    }
    for spec in all_specs:
        validate_feature_alignment(
            spec,
            lowrank_segment_payloads[spec.label],
            segment_payloads[spec.label],
            target_offset_seconds=args.target_offset_seconds,
        )

    native_payloads = {
        spec.label: native_payload_from_baseline(
            spec,
            data_path=args.data_path,
            baseline_payload=segment_payloads[spec.label],
            native_frequency=native_frequency,
        )
        for spec in all_specs
    }
    unit_mask = finite_unit_mask([native_payloads[spec.label] for spec in all_specs])

    prediction_features = {
        spec.label: {
            "features": native_payloads[spec.label]["preds"][:, unit_mask].astype(
                np.float32
            )
        }
        for spec in all_specs
    }
    lowrank_features = {
        spec.label: native_lowrank_features(
            spec,
            lowrank_segment_payloads[spec.label],
            native_payloads[spec.label],
            native_frequency=native_frequency,
            target_offset_seconds=args.target_offset_seconds,
        )
        for spec in all_specs
    }

    prediction_result = fit_and_score(
        train_specs,
        val_specs,
        prediction_features,
        native_payloads,
        unit_mask,
        alphas,
    )
    lowrank_result = fit_and_score(
        train_specs,
        val_specs,
        lowrank_features,
        native_payloads,
        unit_mask,
        alphas,
    )
    train_baseline = score_baseline(train_specs, native_payloads, unit_mask)
    val_baseline = score_baseline(val_specs, native_payloads, unit_mask)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_native_tr_subject_adaptation",
        "checkpoint": args.checkpoint,
        "features": features,
        "subject": sorted({spec.subject for spec in all_specs}),
        "movie": sorted({spec.movie for spec in all_specs}),
        "metric_space": "native_fmri_tr_projected_schaefer1000_parcel",
        "native_frequency_hz": native_frequency,
        "target_offset_seconds": float(args.target_offset_seconds),
        "split_policy": "episode-disjoint available-label local proxy",
        "train_runs": [_jsonable(spec.__dict__) for spec in train_specs],
        "val_runs": [_jsonable(spec.__dict__) for spec in val_specs],
        "n_train_rows": int(
            sum(native_payloads[spec.label]["finite_rows"].sum() for spec in train_specs)
        ),
        "n_val_rows": int(
            sum(native_payloads[spec.label]["finite_rows"].sum() for spec in val_specs)
        ),
        "n_scored_units_common_with_baseline": int(unit_mask.sum()),
        "alpha_grid": alphas,
        "train_baseline": train_baseline,
        "val_baseline": val_baseline,
        "prediction_ridge": {
            **prediction_result,
            "delta_val_vs_baseline": float(
                prediction_result["val_score"]["mean"] - val_baseline["mean"]
            ),
        },
        "lowrank_readout": {
            **lowrank_result,
            "delta_val_vs_baseline": float(
                lowrank_result["val_score"]["mean"] - val_baseline["mean"]
            ),
            "delta_val_vs_prediction_ridge": float(
                lowrank_result["val_score"]["mean"]
                - prediction_result["val_score"]["mean"]
            ),
        },
        "feature_artifacts": {
            label: {
                "path": payload["path"],
                "segment_shape": list(payload["features"].shape),
                "native_shape": list(lowrank_features[label]["features"].shape),
            }
            for label, payload in lowrank_segment_payloads.items()
        },
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy on native fMRI rows. This does not use "
            "Friends S7/OOD hidden labels and is not an official leaderboard score."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--features", nargs="+", default=["text", "audio", "video"])
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("cache/ablation_results/readout_features"),
    )
    parser.add_argument("--target-offset-seconds", type=float, default=5.0)
    parser.add_argument("--train-run", nargs="+", required=True)
    parser.add_argument("--val-run", nargs="+", required=True)
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/tribe_native_tr_subject_adaptation.json"),
    )
    args = parser.parse_args()

    result = run_native_adaptation(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
