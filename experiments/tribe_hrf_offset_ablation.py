"""Evaluate fixed HRF target offsets for exported public TRIBE predictions.

This is a local available-label ablation, not a leaderboard shortcut. The
scientific question is whether the public TRIBE predictions are better aligned
to the subject parcels at a nearby hemodynamic offset than the default 5 s.
Offset selection must be done on training chunks and reported on held-out chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.eval_pretrained_tribe_algonauts import (
    DEFAULT_DATA,
    load_parcel_data,
    resample_parcels_to_segments,
    vertexwise_pearson,
)
from tribev2.studies.algonauts2025 import Algonauts2025


ParcelLoader = Callable[[Path, str, str, str], np.ndarray]


@dataclass(frozen=True)
class PredictionRun:
    result_path: Path
    arrays_path: Path
    subject: str
    movie: str
    chunks: tuple[str, ...]
    baseline_offset_seconds: float
    preds: np.ndarray
    segment_starts: np.ndarray


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


def load_prediction_run(result_path: Path) -> PredictionRun:
    result = json.loads(result_path.read_text())
    arrays_path = Path(result["arrays_path"])
    if not arrays_path.is_absolute():
        arrays_path = result_path.parent.parent.parent / arrays_path
        if not arrays_path.exists():
            arrays_path = Path(result["arrays_path"])
    arrays = np.load(arrays_path)
    sample_times = arrays["sample_times"].astype(np.float32)
    baseline_offset = float(result.get("target_offset_seconds", 5.0))
    return PredictionRun(
        result_path=result_path,
        arrays_path=arrays_path,
        subject=str(result["subject"]),
        movie=str(result["movie"]),
        chunks=tuple(str(chunk) for chunk in result["chunks"]),
        baseline_offset_seconds=baseline_offset,
        preds=arrays["preds"].astype(np.float32),
        segment_starts=sample_times - baseline_offset,
    )


def targets_for_offset(
    run: PredictionRun,
    data_path: Path,
    offset_seconds: float,
    native_frequency: float,
    parcel_loader: ParcelLoader = load_parcel_data,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample chunk parcel targets at segment starts plus offset."""
    target = np.full_like(run.preds, np.nan, dtype=np.float32)
    finite_rows = np.zeros(run.preds.shape[0], dtype=bool)
    time_cursor = 0.0
    for chunk in run.chunks:
        parcels = parcel_loader(data_path, run.subject, run.movie, chunk).astype(np.float32)
        duration = parcels.shape[0] / native_frequency
        sample_times = run.segment_starts + float(offset_seconds)
        in_chunk = (sample_times >= time_cursor) & (sample_times < time_cursor + duration)
        if in_chunk.any():
            local_times = sample_times[in_chunk] - time_cursor
            target[in_chunk] = resample_parcels_to_segments(
                parcels,
                native_frequency=native_frequency,
                sample_times=local_times,
            )
            finite_rows[in_chunk] = np.isfinite(target[in_chunk]).any(axis=1)
        time_cursor += duration
    return target, finite_rows


def score_runs_at_offset(
    runs: list[PredictionRun],
    data_path: Path,
    offset_seconds: float,
    native_frequency: float,
    parcel_loader: ParcelLoader = load_parcel_data,
) -> dict[str, Any]:
    preds_parts = []
    target_parts = []
    finite_parts = []
    per_run = []
    for run in runs:
        target, finite_rows = targets_for_offset(
            run,
            data_path=data_path,
            offset_seconds=offset_seconds,
            native_frequency=native_frequency,
            parcel_loader=parcel_loader,
        )
        metrics = vertexwise_pearson(run.preds[finite_rows], target[finite_rows])
        per_run.append(
            {
                "result_path": str(run.result_path),
                "chunks": list(run.chunks),
                "mean": metrics["mean"],
                "median": metrics["median"],
                "n_vertices_finite": metrics["n_vertices_finite"],
                "n_finite_rows": int(finite_rows.sum()),
            }
        )
        preds_parts.append(run.preds)
        target_parts.append(target)
        finite_parts.append(finite_rows)

    preds = np.concatenate(preds_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    finite_rows = np.concatenate(finite_parts, axis=0)
    metrics = vertexwise_pearson(preds[finite_rows], target[finite_rows])
    return {
        "offset_seconds": float(offset_seconds),
        "mean": metrics["mean"],
        "median": metrics["median"],
        "n_vertices_finite": metrics["n_vertices_finite"],
        "n_finite_rows": int(finite_rows.sum()),
        "per_run": per_run,
    }


def evaluate_offsets(
    runs: list[PredictionRun],
    offsets: list[float],
    data_path: Path,
    native_frequency: float,
    parcel_loader: ParcelLoader = load_parcel_data,
) -> list[dict[str, Any]]:
    return [
        score_runs_at_offset(
            runs,
            data_path=data_path,
            offset_seconds=offset,
            native_frequency=native_frequency,
            parcel_loader=parcel_loader,
        )
        for offset in offsets
    ]


def select_best_offset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if row["mean"] is not None and math.isfinite(row["mean"])]
    if not finite:
        raise ValueError("No finite offset scores available")
    return max(finite, key=lambda row: (row["mean"], -abs(row["offset_seconds"] - 5.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-results", nargs="+", type=Path, required=True)
    parser.add_argument("--val-results", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--offsets",
        nargs="+",
        type=float,
        default=[3.0, 4.0, 5.0, 6.0, 7.0],
        help="Predeclared HRF target offsets to test, in seconds.",
    )
    parser.add_argument("--baseline-offset", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/tribe_hrf_offset_ablation.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    study = Algonauts2025(path=args.data_path)
    native_frequency = float(study._FREQUENCY)
    train_runs = [load_prediction_run(path) for path in args.train_results]
    val_runs = [load_prediction_run(path) for path in args.val_results]

    train_rows = evaluate_offsets(
        train_runs,
        offsets=args.offsets,
        data_path=args.data_path,
        native_frequency=native_frequency,
    )
    best_train = select_best_offset(train_rows)
    val_selected = score_runs_at_offset(
        val_runs,
        data_path=args.data_path,
        offset_seconds=best_train["offset_seconds"],
        native_frequency=native_frequency,
    )
    val_baseline = score_runs_at_offset(
        val_runs,
        data_path=args.data_path,
        offset_seconds=args.baseline_offset,
        native_frequency=native_frequency,
    )
    train_baseline = next(
        row for row in train_rows if row["offset_seconds"] == float(args.baseline_offset)
    )

    delta_val = (
        float(val_selected["mean"] - val_baseline["mean"])
        if val_selected["mean"] is not None and val_baseline["mean"] is not None
        else None
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_hrf_offset_ablation",
        "checkpoint": "facebook/tribev2",
        "features": ["text", "audio", "video"],
        "metric_space": "approx_parcel",
        "offsets_seconds": args.offsets,
        "baseline_offset_seconds": args.baseline_offset,
        "selection_policy": "select highest train mean Pearson; report selected offset on held-out validation chunks",
        "train_results": [str(path) for path in args.train_results],
        "val_results": [str(path) for path in args.val_results],
        "train_scores": train_rows,
        "best_train": best_train,
        "train_baseline": train_baseline,
        "val_selected": val_selected,
        "val_baseline": val_baseline,
        "delta_val_selected_vs_baseline": delta_val,
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy only. Public TRIBE may have seen this "
            "train distribution, and Friends S7/OOD official comparison still "
            "requires Codabench."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
