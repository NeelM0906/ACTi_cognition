"""Calibrate public TRIBE parcel predictions on available Algonauts labels.

This is a local proxy ablation. It does not use hidden Friends S7/OOD labels.
The intended use is to test whether a fixed, subject-specific temporal
calibration layer can improve held-out parcel Pearson over raw public TRIBE
predictions under the same local prediction/target arrays.
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "preds": data["preds"].astype(np.float32),
        "target": data["target"].astype(np.float32),
        "sample_times": data["sample_times"].astype(np.float32),
        "finite_rows": data["finite_rows"].astype(bool),
    }


def _concat_arrays(paths: list[Path]) -> dict[str, np.ndarray]:
    loaded = [_load_npz(path) for path in paths]
    return {
        "preds": np.concatenate([item["preds"] for item in loaded], axis=0),
        "target": np.concatenate([item["target"] for item in loaded], axis=0),
        "sample_times": np.concatenate(
            [item["sample_times"] for item in loaded], axis=0
        ),
        "finite_rows": np.concatenate([item["finite_rows"] for item in loaded], axis=0),
    }


def _split_rows(n_rows: int, train_frac: float, gap: int) -> tuple[np.ndarray, np.ndarray]:
    train_end = int(round(n_rows * train_frac))
    val_start = min(n_rows, train_end + gap)
    train = np.arange(0, train_end, dtype=np.int64)
    val = np.arange(val_start, n_rows, dtype=np.int64)
    if len(train) == 0 or len(val) == 0:
        raise ValueError(
            f"Empty split: n_rows={n_rows}, train_frac={train_frac}, gap={gap}"
        )
    return train, val


def _lagged_design(series: np.ndarray, lags: list[int]) -> np.ndarray:
    n_rows, n_units = series.shape
    blocks = []
    for lag in lags:
        shifted = np.full_like(series, np.nan)
        if lag < 0:
            shifted[:lag] = series[-lag:]
        elif lag > 0:
            shifted[lag:] = series[:-lag]
        else:
            shifted = series.copy()
        blocks.append(shifted)
    return np.stack(blocks, axis=2).reshape(n_rows, n_units, len(lags))


def _fit_temporal_ridge(
    preds: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    lags: list[int],
    alpha: float,
) -> np.ndarray:
    design = _lagged_design(preds, lags)
    calibrated = np.full_like(preds, np.nan, dtype=np.float32)
    eye = np.eye(len(lags) + 1, dtype=np.float64)
    eye[-1, -1] = 0.0

    for unit in range(preds.shape[1]):
        x_train = design[train_rows, unit, :].astype(np.float64)
        y_train = target[train_rows, unit].astype(np.float64)
        train_mask = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
        if train_mask.sum() <= len(lags) + 2:
            continue
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]

        means = x_train.mean(axis=0, keepdims=True)
        stds = x_train.std(axis=0, keepdims=True)
        stds[stds < 1e-6] = 1.0
        x_train = (x_train - means) / stds
        x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
        lhs = x_train.T @ x_train + alpha * eye
        rhs = x_train.T @ y_train
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        x_val = design[val_rows, unit, :].astype(np.float64)
        val_mask = np.isfinite(x_val).all(axis=1)
        if not val_mask.any():
            continue
        x_val_scaled = (x_val[val_mask] - means) / stds
        x_val_scaled = np.concatenate(
            [x_val_scaled, np.ones((x_val_scaled.shape[0], 1))], axis=1
        )
        calibrated[val_rows[val_mask], unit] = (x_val_scaled @ coef).astype(np.float32)
    return calibrated


def _lagged_global_design(series: np.ndarray, lags: list[int]) -> np.ndarray:
    design = _lagged_design(series, lags)
    return design.reshape(series.shape[0], series.shape[1] * len(lags))


def _fit_global_ridge(
    preds: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    lags: list[int],
    alpha: float,
) -> np.ndarray:
    """Fit one ridge adapter from lagged predicted parcels to target parcels.

    This tests whether the public checkpoint has useful spatially distributed
    signal that is misaligned with the local subject parcel targets. It keeps
    evaluation fair by only scoring units that the raw baseline can score.
    """

    design = _lagged_global_design(preds, lags).astype(np.float64)
    train_valid_rows = _valid_rows_for_lags(len(preds), train_rows, lags)
    val_valid_rows = _valid_rows_for_lags(len(preds), val_rows, lags)
    if len(train_valid_rows) == 0 or len(val_valid_rows) == 0:
        raise ValueError("No valid train/validation rows after lag trimming")

    baseline_units = (
        np.isfinite(preds[val_valid_rows]).all(axis=0)
        & np.isfinite(target[val_valid_rows]).all(axis=0)
        & np.isfinite(target[train_valid_rows]).all(axis=0)
    )
    if not baseline_units.any():
        raise ValueError("No finite baseline units available for global ridge")

    x_train = design[train_valid_rows]
    x_val = design[val_valid_rows]
    feature_mask = np.isfinite(x_train).all(axis=0) & np.isfinite(x_val).all(axis=0)
    if not feature_mask.any():
        raise ValueError("No finite lagged prediction features available")

    x_train = x_train[:, feature_mask]
    x_val = x_val[:, feature_mask]
    means = x_train.mean(axis=0, keepdims=True)
    stds = x_train.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    x_train = (x_train - means) / stds
    x_val = (x_val - means) / stds

    y_train = target[train_valid_rows][:, baseline_units].astype(np.float64)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_train = y_train - y_mean

    kernel = x_train @ x_train.T
    kernel.flat[:: kernel.shape[0] + 1] += alpha
    try:
        dual = np.linalg.solve(kernel, y_train)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(kernel, y_train, rcond=None)[0]

    y_pred = x_val @ x_train.T @ dual + y_mean
    calibrated = np.full_like(preds, np.nan, dtype=np.float32)
    unit_indices = np.flatnonzero(baseline_units)
    calibrated[np.ix_(val_valid_rows, unit_indices)] = y_pred.astype(np.float32)
    return calibrated


def _valid_rows_for_lags(n_rows: int, rows: np.ndarray, lags: list[int]) -> np.ndarray:
    valid = np.ones(len(rows), dtype=bool)
    for lag in lags:
        shifted = rows - lag
        valid &= (shifted >= 0) & (shifted < n_rows)
    return rows[valid]


def _finite_subset(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    finite_rows = arrays["finite_rows"] & np.isfinite(arrays["target"]).any(axis=1)
    return {
        "preds": arrays["preds"][finite_rows],
        "target": arrays["target"][finite_rows],
        "sample_times": arrays["sample_times"][finite_rows],
        "finite_rows": arrays["finite_rows"][finite_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--train-arrays", nargs="+", type=Path)
    parser.add_argument("--val-arrays", nargs="+", type=Path)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--lags", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument(
        "--lags-json",
        help="JSON list of integer lags. Overrides --lags and avoids shell parsing issues.",
    )
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--adapter",
        choices=["per-parcel", "global-ridge"],
        default="per-parcel",
        help=(
            "per-parcel fits an independent temporal ridge per parcel; "
            "global-ridge fits one spatiotemporal ridge adapter across parcels."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/tribe_calibration_ablation.json"),
    )
    args = parser.parse_args()
    if args.lags_json:
        args.lags = [int(lag) for lag in json.loads(args.lags_json)]

    start = time.perf_counter()
    if args.train_arrays or args.val_arrays:
        if not args.train_arrays or not args.val_arrays:
            raise ValueError("--train-arrays and --val-arrays must be provided together")
        train_arrays = _finite_subset(_concat_arrays(args.train_arrays))
        val_arrays = _finite_subset(_concat_arrays(args.val_arrays))
        preds = np.concatenate([train_arrays["preds"], val_arrays["preds"]], axis=0)
        target = np.concatenate([train_arrays["target"], val_arrays["target"]], axis=0)
        sample_times = np.concatenate(
            [train_arrays["sample_times"], val_arrays["sample_times"]], axis=0
        )
        train_local = np.arange(train_arrays["preds"].shape[0], dtype=np.int64)
        val_local = np.arange(
            train_arrays["preds"].shape[0], preds.shape[0], dtype=np.int64
        )
        source_arrays = {
            "train": args.train_arrays,
            "val": args.val_arrays,
            "mode": "explicit_train_val_arrays",
        }
    elif args.arrays:
        arrays = _finite_subset(_load_npz(args.arrays))
        preds = arrays["preds"]
        target = arrays["target"]
        sample_times = arrays["sample_times"]
        train_local, val_local = _split_rows(len(preds), args.train_frac, args.gap)
        source_arrays = {"arrays": args.arrays, "mode": "single_array_time_split"}
    else:
        raise ValueError("Provide either --arrays or --train-arrays/--val-arrays")

    if args.adapter == "global-ridge":
        calibrated = _fit_global_ridge(
            preds=preds,
            target=target,
            train_rows=train_local,
            val_rows=val_local,
            lags=args.lags,
            alpha=args.alpha,
        )
    else:
        calibrated = _fit_temporal_ridge(
            preds=preds,
            target=target,
            train_rows=train_local,
            val_rows=val_local,
            lags=args.lags,
            alpha=args.alpha,
        )
    metric_val_local = _valid_rows_for_lags(len(preds), val_local, args.lags)
    baseline = vertexwise_pearson(preds[metric_val_local], target[metric_val_local])
    candidate = vertexwise_pearson(
        calibrated[metric_val_local], target[metric_val_local]
    )
    seconds = time.perf_counter() - start

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_arrays": source_arrays,
        "train_frac": args.train_frac,
        "gap": args.gap,
        "lags": args.lags,
        "alpha": args.alpha,
        "adapter": args.adapter,
        "n_rows_total": int(preds.shape[0]),
        "n_rows_finite": int(preds.shape[0]),
        "n_train_rows": int(len(train_local)),
        "n_val_rows": int(len(val_local)),
        "n_metric_val_rows": int(len(metric_val_local)),
        "train_time_range": [
            float(sample_times[train_local[0]]),
            float(sample_times[train_local[-1]]),
        ],
        "val_time_range": [
            float(sample_times[val_local[0]]),
            float(sample_times[val_local[-1]]),
        ],
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "delta_mean": candidate["mean"] - baseline["mean"],
        "delta_median": candidate["median"] - baseline["median"],
        "runtime_seconds": seconds,
        "caveat": (
            "Local calibration proxy on available labels only; not an official "
            "Algonauts leaderboard result."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
