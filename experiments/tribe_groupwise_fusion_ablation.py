"""Network-wise late fusion over public TRIBE modality predictions.

This is a regularized version of parcel-wise fusion: modality weights are shared
within Schaefer/Yeo functional networks instead of selected independently for
every parcel. It tests whether the small parcel-wise fusion gain reflects
coherent cortical specialization or mostly per-parcel noise.
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
from nilearn import datasets

from experiments.eval_pretrained_tribe_algonauts import vertexwise_pearson
from experiments.tribe_late_fusion_ablation import (
    aligned_dataset,
    candidate_weights,
    fuse_predictions,
    parse_condition_args,
    parse_grid_specs,
    score_weights,
)
from experiments.tribe_regionwise_fusion_ablation import columnwise_pearson


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


def schaefer_group_labels(mode: str) -> tuple[np.ndarray, list[str]]:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=1000,
        yeo_networks=7,
        resolution_mm=2,
        verbose=0,
    )
    labels = atlas.labels[1:]
    group_names = []
    groups = []
    for label in labels:
        parts = str(label).split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected Schaefer label: {label}")
        hemi = parts[1]
        network = parts[2]
        key = network if mode == "network" else f"{hemi}_{network}"
        if key not in group_names:
            group_names.append(key)
        groups.append(group_names.index(key))
    return np.asarray(groups, dtype=np.int64), group_names


def fit_groupwise_weights(
    train_preds: np.ndarray,
    train_target: np.ndarray,
    conditions: list[str],
    candidates: list[dict[str, float]],
    groups: np.ndarray,
) -> np.ndarray:
    if train_preds.shape[2] != groups.shape[0]:
        raise ValueError(f"Unit/group mismatch: {train_preds.shape[2]} vs {groups.shape[0]}")
    candidate_corrs = []
    for weights in candidates:
        fused = fuse_predictions(train_preds, conditions, weights)
        candidate_corrs.append(columnwise_pearson(fused, train_target))
    corr_matrix = np.stack(candidate_corrs, axis=0)
    weights_by_unit = np.zeros((groups.shape[0], len(conditions)), dtype=np.float32)
    for group in np.unique(groups):
        units = np.flatnonzero(groups == group)
        group_scores = np.nanmean(corr_matrix[:, units], axis=1)
        if not np.isfinite(group_scores).any():
            continue
        idx = int(np.nanargmax(group_scores))
        weights = candidates[idx]
        weights_by_unit[units] = [weights.get(condition, 0.0) for condition in conditions]
    return weights_by_unit


def apply_groupwise_weights(preds: np.ndarray, weights_by_unit: np.ndarray) -> np.ndarray:
    return np.einsum("rcu,uc->ru", preds, weights_by_unit, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-condition", nargs="+", action="append", required=True)
    parser.add_argument("--val-condition", nargs="+", action="append", required=True)
    parser.add_argument("--grid", action="append", required=True)
    parser.add_argument("--group-mode", choices=["network", "hemi-network"], default="hemi-network")
    parser.add_argument("--full-condition", default="full")
    parser.add_argument("--time-tolerance-seconds", type=float, default=0.06)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/tribe_groupwise_fusion_ablation.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    train_paths = parse_condition_args(args.train_condition)
    val_paths = parse_condition_args(args.val_condition)
    conditions, train_preds, train_target = aligned_dataset(
        train_paths, tolerance_seconds=args.time_tolerance_seconds
    )
    val_conditions, val_preds, val_target = aligned_dataset(
        val_paths, tolerance_seconds=args.time_tolerance_seconds
    )
    if conditions != val_conditions:
        raise ValueError(f"Condition order mismatch: {conditions} vs {val_conditions}")

    groups, group_names = schaefer_group_labels(args.group_mode)
    candidates = candidate_weights(conditions, parse_grid_specs(args.grid))
    weights_by_unit = fit_groupwise_weights(
        train_preds, train_target, conditions, candidates, groups
    )
    val_groupwise_pred = apply_groupwise_weights(val_preds, weights_by_unit)
    val_groupwise = vertexwise_pearson(val_groupwise_pred, val_target)

    full_weights = {condition: 0.0 for condition in conditions}
    full_weights[args.full_condition] = 1.0
    val_full = score_weights(val_preds, val_target, conditions, full_weights)
    delta_val = (
        float(val_groupwise["mean"] - val_full["mean"])
        if val_groupwise["mean"] is not None and val_full["mean"] is not None
        else None
    )

    group_summary = {}
    for idx, name in enumerate(group_names):
        units = np.flatnonzero(groups == idx)
        group_summary[name] = {
            condition: float(np.nanmean(weights_by_unit[units, cidx]))
            for cidx, condition in enumerate(conditions)
        }

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_groupwise_fusion_ablation",
        "conditions": conditions,
        "group_mode": args.group_mode,
        "n_groups": len(group_names),
        "grids": args.grid,
        "time_tolerance_seconds": args.time_tolerance_seconds,
        "n_candidates": len(candidates),
        "train_shape": list(train_preds.shape),
        "val_shape": list(val_preds.shape),
        "group_weight_summary": group_summary,
        "val_groupwise": {
            "mean": val_groupwise["mean"],
            "median": val_groupwise["median"],
            "n_vertices_finite": val_groupwise["n_vertices_finite"],
        },
        "val_full": val_full,
        "delta_val_groupwise_vs_full": delta_val,
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy only. Group-wise fusion weights were "
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
