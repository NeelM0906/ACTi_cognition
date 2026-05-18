"""Region-wise convex late fusion over public TRIBE modality predictions.

This is a constrained output-space algorithm: each parcel selects a convex
combination of a small set of modality predictions on train chunks, then the
selected parcel-wise weights are evaluated on held-out chunks. It tests whether
functional specialization across cortex can improve over the public full-fusion
checkpoint without using hidden labels.
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

from experiments.eval_pretrained_tribe_algonauts import vertexwise_pearson
from experiments.tribe_late_fusion_ablation import (
    aligned_dataset,
    candidate_weights,
    fuse_predictions,
    parse_condition_args,
    parse_grid_specs,
    score_weights,
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


def columnwise_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    valid_cols = np.isfinite(pred).all(axis=0) & np.isfinite(target).all(axis=0)
    corr = np.full(pred.shape[1], np.nan, dtype=np.float64)
    if not valid_cols.any():
        return corr
    x = pred[:, valid_cols].astype(np.float64)
    y = target[:, valid_cols].astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    ok = denom > 0
    local = np.full(x.shape[1], np.nan, dtype=np.float64)
    local[ok] = (x[:, ok] * y[:, ok]).sum(axis=0) / denom[ok]
    corr[np.flatnonzero(valid_cols)] = local
    return corr


def fit_regionwise_weights(
    train_preds: np.ndarray,
    train_target: np.ndarray,
    conditions: list[str],
    candidates: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    corrs = []
    for weights in candidates:
        fused = fuse_predictions(train_preds, conditions, weights)
        corrs.append(columnwise_pearson(fused, train_target))
    corr_matrix = np.stack(corrs, axis=0)
    filled = np.where(np.isfinite(corr_matrix), corr_matrix, -np.inf)
    best_idx = np.argmax(filled, axis=0)
    no_valid = ~np.isfinite(filled).any(axis=0)
    best_idx[no_valid] = -1

    weights_by_unit = np.zeros((train_preds.shape[2], len(conditions)), dtype=np.float32)
    for unit, idx in enumerate(best_idx):
        if idx < 0:
            continue
        weights = candidates[int(idx)]
        weights_by_unit[unit] = [weights.get(condition, 0.0) for condition in conditions]
    selected_corr = np.full(train_preds.shape[2], np.nan, dtype=np.float64)
    valid_units = best_idx >= 0
    selected_corr[valid_units] = corr_matrix[best_idx[valid_units], np.flatnonzero(valid_units)]
    return weights_by_unit, selected_corr


def apply_regionwise_weights(preds: np.ndarray, weights_by_unit: np.ndarray) -> np.ndarray:
    if preds.shape[1] != weights_by_unit.shape[1]:
        raise ValueError(
            f"Condition mismatch: predictions have {preds.shape[1]} conditions, "
            f"weights have {weights_by_unit.shape[1]}"
        )
    return np.einsum("rcu,uc->ru", preds, weights_by_unit, optimize=True)


def summarize_weights(weights_by_unit: np.ndarray, conditions: list[str]) -> dict[str, Any]:
    nonzero_counts = {
        condition: int(np.count_nonzero(weights_by_unit[:, idx] > 1e-6))
        for idx, condition in enumerate(conditions)
    }
    mean_weights = {
        condition: float(np.nanmean(weights_by_unit[:, idx]))
        for idx, condition in enumerate(conditions)
    }
    return {"nonzero_counts": nonzero_counts, "mean_weights": mean_weights}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-condition", nargs="+", action="append", required=True)
    parser.add_argument("--val-condition", nargs="+", action="append", required=True)
    parser.add_argument("--grid", action="append", required=True)
    parser.add_argument("--full-condition", default="full")
    parser.add_argument("--time-tolerance-seconds", type=float, default=0.06)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/tribe_regionwise_fusion_ablation.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    train_paths = parse_condition_args(args.train_condition)
    val_paths = parse_condition_args(args.val_condition)
    if set(train_paths) != set(val_paths):
        raise ValueError("Train and validation conditions must match")
    conditions, train_preds, train_target = aligned_dataset(
        train_paths, tolerance_seconds=args.time_tolerance_seconds
    )
    val_conditions, val_preds, val_target = aligned_dataset(
        val_paths, tolerance_seconds=args.time_tolerance_seconds
    )
    if conditions != val_conditions:
        raise ValueError(f"Condition order mismatch: {conditions} vs {val_conditions}")

    candidates = candidate_weights(conditions, parse_grid_specs(args.grid))
    weights_by_unit, selected_train_corr = fit_regionwise_weights(
        train_preds, train_target, conditions, candidates
    )
    val_regionwise_pred = apply_regionwise_weights(val_preds, weights_by_unit)
    val_regionwise = vertexwise_pearson(val_regionwise_pred, val_target)

    full_weights = {condition: 0.0 for condition in conditions}
    full_weights[args.full_condition] = 1.0
    val_full = score_weights(val_preds, val_target, conditions, full_weights)
    delta_val = (
        float(val_regionwise["mean"] - val_full["mean"])
        if val_regionwise["mean"] is not None and val_full["mean"] is not None
        else None
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_regionwise_fusion_ablation",
        "conditions": conditions,
        "grids": args.grid,
        "time_tolerance_seconds": args.time_tolerance_seconds,
        "n_candidates": len(candidates),
        "train_shape": list(train_preds.shape),
        "val_shape": list(val_preds.shape),
        "train_selected_mean_unit_corr": float(np.nanmean(selected_train_corr)),
        "train_selected_median_unit_corr": float(np.nanmedian(selected_train_corr)),
        "weight_summary": summarize_weights(weights_by_unit, conditions),
        "val_regionwise": {
            "mean": val_regionwise["mean"],
            "median": val_regionwise["median"],
            "n_vertices_finite": val_regionwise["n_vertices_finite"],
        },
        "val_full": val_full,
        "delta_val_regionwise_vs_full": delta_val,
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy only. Parcel-wise fusion weights were "
            "selected on train chunks and evaluated on held-out available labels; "
            "no hidden Friends S7/OOD labels were used."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
