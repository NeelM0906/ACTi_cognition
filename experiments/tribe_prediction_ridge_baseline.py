"""Supervised ridge baseline over public TRIBE parcel predictions.

This is the matched calibration control for the frozen-low-rank readout
ablation. It uses the same train/validation run protocol and alpha selection,
but its input features are only the public average-head parcel predictions.
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

from experiments.tribe_frozen_readout_ablation import (
    finite_unit_mask,
    fit_ridge_readout,
    load_baseline_run,
    load_baseline_metadata,
    parse_run_spec,
    predict_ridge,
    prepare_xy,
    score_baseline,
    select_alpha,
    validate_baseline_metadata,
    validate_disjoint_runs,
)


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


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    train_specs = [parse_run_spec(text) for text in args.train_run]
    val_specs = [parse_run_spec(text) for text in args.val_run]
    validate_disjoint_runs(train_specs, val_specs)
    all_specs = train_specs + val_specs
    baseline_payloads = {spec.label: load_baseline_run(spec) for spec in all_specs}
    baseline_metadata = {spec.label: load_baseline_metadata(spec) for spec in all_specs}
    for spec in all_specs:
        validate_baseline_metadata(
            spec,
            baseline_metadata[spec.label],
            baseline_payloads[spec.label],
            expected_checkpoint=args.checkpoint,
            expected_features=list(args.features),
            expected_metric_space=args.metric_space,
            expected_target_offset_seconds=args.target_offset_seconds,
        )
    unit_mask = finite_unit_mask([baseline_payloads[spec.label] for spec in all_specs])
    feature_payloads = {}
    for spec in all_specs:
        run = baseline_payloads[spec.label]
        feature_payloads[spec.label] = {
            "features": run["preds"][:, unit_mask].astype(np.float32)
        }

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
    val_calibrated = predict_ridge(x_val, fitted)
    train_calibrated = predict_ridge(x_train, fitted)

    from experiments.eval_pretrained_tribe_algonauts import vertexwise_pearson

    val_calibrated_score = vertexwise_pearson(val_calibrated, y_val)
    train_calibrated_score = vertexwise_pearson(train_calibrated, y_train)
    val_baseline = score_baseline(val_specs, baseline_payloads, unit_mask)
    train_baseline = score_baseline(train_specs, baseline_payloads, unit_mask)
    delta_val = float(val_calibrated_score["mean"] - val_baseline["mean"])

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_prediction_ridge_baseline",
        "baseline": "public facebook/tribev2 full average-head parcel predictions",
        "candidate": "supervised ridge calibration over public average-head parcel predictions",
        "checkpoint": args.checkpoint,
        "features": list(args.features),
        "metric_space": "approx_projected_schaefer1000_parcel_1hz_interpolated_targets",
        "split_policy": "episode-disjoint available-label local proxy",
        "target_offset_seconds": float(args.target_offset_seconds),
        "train_runs": [_jsonable(spec.__dict__) for spec in train_specs],
        "val_runs": [_jsonable(spec.__dict__) for spec in val_specs],
        "n_train_rows": int(x_train.shape[0]),
        "n_val_rows": int(x_val.shape[0]),
        "n_scored_units_common_with_baseline": int(unit_mask.sum()),
        "alpha_grid": [float(alpha) for alpha in args.alpha],
        "selected_alpha": selected_alpha,
        "inner_alpha_selection": alpha_result,
        "train_calibrated": train_calibrated_score,
        "train_baseline": train_baseline,
        "val_calibrated": val_calibrated_score,
        "val_baseline": val_baseline,
        "delta_val_calibrated_vs_baseline": delta_val,
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy only. This is a supervised calibration "
            "control for the frozen low-rank readout, not a hidden leaderboard score."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", nargs="+", required=True)
    parser.add_argument("--val-run", nargs="+", required=True)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--features", nargs="+", default=["text", "audio", "video"])
    parser.add_argument("--metric-space", default="parcel")
    parser.add_argument("--target-offset-seconds", type=float, default=5.0)
    parser.add_argument("--alpha", nargs="+", type=float, default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--out", type=Path, default=Path("cache/ablation_results/tribe_prediction_ridge_baseline.json"))
    args = parser.parse_args()

    result = run_baseline(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
