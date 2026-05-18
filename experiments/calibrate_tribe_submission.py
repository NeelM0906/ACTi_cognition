"""Apply subject-specific prediction-ridge calibration to submission files.

This is the official-format candidate path for the current best local result:
fit a supervised ridge map from public TRIBE parcel predictions to available
training fMRI labels, then apply that map to hidden-stimulus raw predictions.
Hidden Friends S7/OOD labels are never loaded here; the input submission file
contains predictions only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.eval_pretrained_tribe_algonauts import DEFAULT_DATA, vertexwise_pearson
from experiments.tribe_frozen_readout_ablation import (
    RunSpec,
    _json_sidecar_path,
    episode_identity,
    finite_unit_mask,
    fit_ridge_readout,
    load_baseline_metadata,
    load_baseline_run,
    parse_run_spec,
    predict_ridge,
    prepare_xy,
    validate_baseline_metadata,
)
from experiments.tribe_native_tr_subject_adaptation import native_payload_from_baseline
from tribev2.studies.algonauts2025 import Algonauts2025


DEFAULT_BASELINE_DIR = Path("cache/ablation_results")
DEFAULT_TRAIN_CHUNKS = ("e01a", "e01b", "e02a", "e02b")
DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


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


def _zip_file(source: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source, arcname=source.name)


def default_baseline_npz(
    baseline_dir: Path,
    subject: str,
    movie: str,
    chunk: str,
) -> Path:
    """Find the local public-TRIBE baseline array for a subject/chunk."""
    candidates = [
        baseline_dir
        / f"pretrained_tribe_text_audio_video_{subject}_{movie}{chunk}_parcel_approx_v1.npz",
        baseline_dir
        / f"pretrained_tribe_text_audio_video_{subject}_{movie}{chunk}_parcel_approx_v2.npz",
    ]
    if subject == "sub-01":
        candidates.extend(
            [
                baseline_dir
                / f"pretrained_tribe_text_audio_video_{movie}{chunk}_parcel_approx_v1.npz",
                baseline_dir
                / f"pretrained_tribe_text_audio_video_{movie}{chunk}_parcel_approx_v2.npz",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No baseline NPZ found for {subject}/{movie}{chunk}; tried "
        f"{[str(path) for path in candidates]}"
    )


def build_default_run_specs(
    baseline_dir: Path,
    subject: str,
    movie: str,
    chunks: list[str],
) -> list[RunSpec]:
    return [
        RunSpec(
            label=chunk,
            subject=subject,
            movie=movie,
            chunk=chunk,
            baseline_npz=default_baseline_npz(baseline_dir, subject, movie, chunk),
        )
        for chunk in chunks
    ]


def load_native_training_payloads(
    specs: list[RunSpec],
    *,
    data_path: Path,
    checkpoint: str,
    features: list[str],
    metric_space: str,
    target_offset_seconds: float,
) -> dict[str, dict[str, np.ndarray]]:
    native_frequency = float(Algonauts2025._FREQUENCY)
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        segment_payload = load_baseline_run(spec)
        metadata = load_baseline_metadata(spec)
        validate_baseline_metadata(
            spec,
            metadata,
            segment_payload,
            expected_checkpoint=checkpoint,
            expected_features=features,
            expected_metric_space=metric_space,
            expected_target_offset_seconds=target_offset_seconds,
        )
        payloads[spec.label] = native_payload_from_baseline(
            spec,
            data_path=data_path,
            baseline_payload=segment_payload,
            native_frequency=native_frequency,
        )
    return payloads


def _feature_payloads_from_baseline(
    specs: list[RunSpec],
    baseline_payloads: dict[str, dict[str, np.ndarray]],
    unit_mask: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        spec.label: {
            "features": baseline_payloads[spec.label]["preds"][:, unit_mask].astype(
                np.float32
            )
        }
        for spec in specs
    }


def select_alpha_by_episode(
    specs: list[RunSpec],
    feature_payloads: dict[str, dict[str, np.ndarray]],
    baseline_payloads: dict[str, dict[str, np.ndarray]],
    unit_mask: np.ndarray,
    alphas: list[float],
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[RunSpec]] = defaultdict(list)
    for spec in specs:
        groups[episode_identity(spec)].append(spec)
    if len(groups) < 2:
        return {"selected_alpha": float(alphas[0]), "inner_scores": []}

    group_items = list(groups.items())
    inner_scores = []
    for alpha in alphas:
        fold_scores = []
        for holdout_group, holdout_specs in group_items:
            train_specs = [
                spec
                for group, group_specs in group_items
                if group != holdout_group
                for spec in group_specs
            ]
            x_train, y_train = prepare_xy(
                train_specs,
                feature_payloads,
                baseline_payloads,
                unit_mask,
            )
            x_val, y_val = prepare_xy(
                holdout_specs,
                feature_payloads,
                baseline_payloads,
                unit_mask,
            )
            fitted = fit_ridge_readout(x_train, y_train, alpha=float(alpha))
            score = vertexwise_pearson(predict_ridge(x_val, fitted), y_val)
            fold_scores.append(
                {
                    "episode": "/".join(holdout_group),
                    "mean_score": float(score["mean"]),
                    "n_val_rows": int(x_val.shape[0]),
                }
            )
        inner_scores.append(
            {
                "alpha": float(alpha),
                "mean_score": float(
                    np.mean([fold["mean_score"] for fold in fold_scores])
                ),
                "fold_scores": fold_scores,
            }
        )
    best = max(inner_scores, key=lambda item: item["mean_score"])
    return {"selected_alpha": float(best["alpha"]), "inner_scores": inner_scores}


def fit_subject_calibrator(
    specs: list[RunSpec],
    *,
    data_path: Path,
    checkpoint: str,
    features: list[str],
    metric_space: str,
    target_offset_seconds: float,
    alphas: list[float],
) -> dict[str, Any]:
    baseline_payloads = load_native_training_payloads(
        specs,
        data_path=data_path,
        checkpoint=checkpoint,
        features=features,
        metric_space=metric_space,
        target_offset_seconds=target_offset_seconds,
    )
    unit_mask = finite_unit_mask([baseline_payloads[spec.label] for spec in specs])
    feature_payloads = _feature_payloads_from_baseline(
        specs, baseline_payloads, unit_mask
    )
    alpha_result = select_alpha_by_episode(
        specs,
        feature_payloads,
        baseline_payloads,
        unit_mask,
        alphas,
    )
    x_train, y_train = prepare_xy(specs, feature_payloads, baseline_payloads, unit_mask)
    fitted = fit_ridge_readout(
        x_train,
        y_train,
        alpha=float(alpha_result["selected_alpha"]),
    )
    train_score = vertexwise_pearson(predict_ridge(x_train, fitted), y_train)
    raw_score = vertexwise_pearson(x_train, y_train)
    return {
        "fitted": fitted,
        "unit_mask": unit_mask,
        "alpha_selection": alpha_result,
        "n_train_rows": int(x_train.shape[0]),
        "n_calibrated_units": int(unit_mask.sum()),
        "train_raw_score": raw_score,
        "train_calibrated_score": train_score,
        "train_run_specs": [_jsonable(spec.__dict__) for spec in specs],
    }


def apply_calibration_to_prediction(
    raw_prediction: np.ndarray,
    fitted: dict[str, np.ndarray],
    unit_mask: np.ndarray,
) -> np.ndarray:
    if raw_prediction.ndim != 2:
        raise ValueError(f"Expected 2-D prediction, got {raw_prediction.shape}")
    if raw_prediction.shape[1] != unit_mask.shape[0]:
        raise ValueError(
            f"Prediction/unit mask mismatch: {raw_prediction.shape[1]} vs {unit_mask.shape[0]}"
        )
    calibrated = raw_prediction.astype(np.float32, copy=True)
    if unit_mask.any():
        calibrated[:, unit_mask] = predict_ridge(
            calibrated[:, unit_mask],
            fitted,
        )
    return calibrated.astype(np.float32)


def write_fitted_calibrator(
    fitted_dir: Path,
    subject: str,
    payload: dict[str, Any],
    manifest_payload: dict[str, Any],
) -> Path:
    fitted_dir.mkdir(parents=True, exist_ok=True)
    path = fitted_dir / f"prediction_ridge_{subject}_native_tr_s01e01e02_v1.npz"
    np.savez_compressed(
        path,
        weights=payload["fitted"]["weights"],
        x_mean=payload["fitted"]["x_mean"],
        y_mean=payload["fitted"]["y_mean"],
        unit_mask=payload["unit_mask"],
        manifest_json=np.asarray(json.dumps(_jsonable(manifest_payload))),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--raw-submission", type=Path, required=True)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--features", nargs="+", default=["text", "audio", "video"])
    parser.add_argument("--metric-space", default="parcel")
    parser.add_argument("--movie", default="s01")
    parser.add_argument("--train-chunks", nargs="+", default=list(DEFAULT_TRAIN_CHUNKS))
    parser.add_argument(
        "--train-run",
        nargs="+",
        help=(
            "Optional explicit training specs label=subject:movie:chunk:path. "
            "If omitted, specs are inferred per raw-submission subject."
        ),
    )
    parser.add_argument(
        "--target-offset-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--fitted-dir",
        type=Path,
        default=Path("cache/submissions/calibrators"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/submissions/tribe_prediction_ridge_calibrated.npy"),
    )
    parser.add_argument("--zip-out", type=Path)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("cache/submissions/tribe_prediction_ridge_calibrated_manifest.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    raw = np.load(args.raw_submission, allow_pickle=True).item()
    if not isinstance(raw, dict):
        raise ValueError("Raw submission must be a nested subject/item dictionary")

    predictions: dict[str, dict[str, np.ndarray]] = {}
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_submission": str(args.raw_submission),
        "checkpoint": args.checkpoint,
        "features": list(args.features),
        "metric_space": "native_fmri_tr_projected_schaefer1000_parcel",
        "target_offset_seconds": float(args.target_offset_seconds),
        "alpha_grid": [float(alpha) for alpha in args.alpha],
        "candidate": "subject-specific prediction-ridge calibration over public TRIBE parcel predictions",
        "caveat": (
            "Fits only available labeled Friends season-1 runs and applies the map "
            "to hidden-stimulus predictions. Hidden Friends S7/OOD labels are not loaded."
        ),
        "subjects": {},
    }

    explicit_specs = [parse_run_spec(text) for text in args.train_run or []]
    specs_by_subject: dict[str, list[RunSpec]] = defaultdict(list)
    for spec in explicit_specs:
        specs_by_subject[spec.subject].append(spec)

    for subject, raw_items in raw.items():
        subject_start = time.perf_counter()
        specs = specs_by_subject.get(subject)
        if not specs:
            specs = build_default_run_specs(
                args.baseline_dir,
                subject,
                args.movie,
                list(args.train_chunks),
            )
        calibrator = fit_subject_calibrator(
            specs,
            data_path=args.data_path,
            checkpoint=args.checkpoint,
            features=list(args.features),
            metric_space=args.metric_space,
            target_offset_seconds=float(args.target_offset_seconds),
            alphas=[float(alpha) for alpha in args.alpha],
        )
        subject_manifest = {
            key: value
            for key, value in calibrator.items()
            if key not in {"fitted", "unit_mask"}
        }
        fitted_path = write_fitted_calibrator(
            args.fitted_dir,
            subject,
            calibrator,
            subject_manifest,
        )
        subject_manifest["fitted_calibrator_path"] = str(fitted_path)
        subject_manifest["items"] = {}

        predictions[subject] = {}
        for item, raw_prediction in raw_items.items():
            item_start = time.perf_counter()
            raw_prediction = np.asarray(raw_prediction, dtype=np.float32)
            calibrated = apply_calibration_to_prediction(
                raw_prediction,
                calibrator["fitted"],
                calibrator["unit_mask"],
            )
            if calibrated.shape != raw_prediction.shape:
                raise ValueError(
                    f"{subject}/{item}: calibrated shape {calibrated.shape} "
                    f"does not match raw {raw_prediction.shape}"
                )
            if not np.isfinite(calibrated).all():
                raise ValueError(f"{subject}/{item}: calibrated prediction has NaN/inf")
            predictions[subject][item] = calibrated.astype(np.float32)
            subject_manifest["items"][item] = {
                "shape": list(calibrated.shape),
                "raw_mean": float(np.mean(raw_prediction)),
                "calibrated_mean": float(np.mean(calibrated)),
                "raw_std": float(np.std(raw_prediction)),
                "calibrated_std": float(np.std(calibrated)),
                "calibrated_units": int(calibrator["unit_mask"].sum()),
                "uncalibrated_units_preserved": int(
                    calibrator["unit_mask"].shape[0] - calibrator["unit_mask"].sum()
                ),
                "item_seconds": time.perf_counter() - item_start,
            }
        subject_manifest["subject_seconds"] = time.perf_counter() - subject_start
        manifest["subjects"][subject] = subject_manifest

    manifest["runtime_seconds"] = time.perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, predictions)
    manifest["out"] = str(args.out)
    if args.zip_out is not None:
        _zip_file(args.out, args.zip_out)
        manifest["zip_out"] = str(args.zip_out)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(_jsonable(manifest), indent=2),
        encoding="utf-8",
    )
    print(args.out)
    print(args.manifest_out)
    print(json.dumps(_jsonable(manifest), indent=2))


if __name__ == "__main__":
    main()
