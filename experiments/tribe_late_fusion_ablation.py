"""Late-fusion ablation over exported public TRIBE modality predictions.

This tests a constrained algorithmic question: can a global convex combination
of public TRIBE modality-specific predictions beat the public full multimodal
prediction on held-out available labels? Weights are selected only on train
chunks and then reported on held-out chunks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.eval_pretrained_tribe_algonauts import vertexwise_pearson


@dataclass(frozen=True)
class ConditionRun:
    condition: str
    result_path: Path
    preds: np.ndarray
    target: np.ndarray
    sample_times: np.ndarray


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


def _resolve_arrays_path(result_path: Path, arrays_path: str) -> Path:
    path = Path(arrays_path)
    if path.exists() or path.is_absolute():
        return path
    repo_relative = result_path.resolve().parents[2] / path
    if repo_relative.exists():
        return repo_relative
    return path


def load_condition_run(condition: str, result_path: Path) -> ConditionRun:
    result = json.loads(result_path.read_text())
    arrays_path = _resolve_arrays_path(result_path, result["arrays_path"])
    arrays = np.load(arrays_path)
    return ConditionRun(
        condition=condition,
        result_path=result_path,
        preds=arrays["preds"].astype(np.float32),
        target=arrays["target"].astype(np.float32),
        sample_times=arrays["sample_times"].astype(np.float32),
    )


def parse_condition_args(items: list[list[str]]) -> dict[str, list[Path]]:
    parsed: dict[str, list[Path]] = {}
    for item in items:
        if len(item) < 2:
            raise ValueError("Each condition needs a name and at least one result path")
        name, *paths = item
        if name in parsed:
            raise ValueError(f"Duplicate condition: {name}")
        parsed[name] = [Path(path) for path in paths]
    return parsed


def common_time_indices(
    runs: list[ConditionRun],
    tolerance_seconds: float = 0.25,
) -> dict[str, np.ndarray]:
    reference = runs[0].sample_times.astype(np.float64)
    matches: dict[str, list[int]] = {run.condition: [] for run in runs}

    other_times = [run.sample_times.astype(np.float64) for run in runs[1:]]
    for ref_idx, time_value in enumerate(reference):
        row = [ref_idx]
        ok = True
        for times in other_times:
            insert = int(np.searchsorted(times, time_value))
            candidates = []
            if insert < len(times):
                candidates.append(insert)
            if insert > 0:
                candidates.append(insert - 1)
            if not candidates:
                ok = False
                break
            best = min(candidates, key=lambda idx: abs(float(times[idx] - time_value)))
            if abs(float(times[best] - time_value)) > tolerance_seconds:
                ok = False
                break
            row.append(best)
        if ok:
            for run, row_idx in zip(runs, row, strict=True):
                matches[run.condition].append(row_idx)

    if not matches[runs[0].condition]:
        raise ValueError("No approximately common sample times across condition runs")
    return {
        condition: np.array(indices, dtype=np.int64)
        for condition, indices in matches.items()
    }


def aligned_dataset(
    condition_paths: dict[str, list[Path]],
    tolerance_seconds: float = 0.25,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    conditions = list(condition_paths)
    lengths = {len(paths) for paths in condition_paths.values()}
    if len(lengths) != 1:
        raise ValueError(f"Conditions must provide the same number of runs: {lengths}")

    pred_blocks = []
    target_blocks = []
    n_runs = lengths.pop()
    for idx in range(n_runs):
        runs = [
            load_condition_run(condition, condition_paths[condition][idx])
            for condition in conditions
        ]
        indices = common_time_indices(runs, tolerance_seconds=tolerance_seconds)
        block_preds = []
        for run in runs:
            block_preds.append(run.preds[indices[run.condition]])
        first = runs[0]
        target = first.target[indices[first.condition]]
        finite_rows = np.isfinite(target).any(axis=1)
        pred_blocks.append(np.stack(block_preds, axis=1)[finite_rows])
        target_blocks.append(target[finite_rows])

    return conditions, np.concatenate(pred_blocks, axis=0), np.concatenate(target_blocks, axis=0)


def convex_grid(condition_names: list[str], step: float) -> list[dict[str, float]]:
    if step <= 0 or step > 1:
        raise ValueError(f"Invalid grid step: {step}")
    n_steps = int(round(1.0 / step))
    if not np.isclose(n_steps * step, 1.0):
        raise ValueError(f"Step must divide 1.0 exactly: {step}")
    weights = []
    for counts in itertools.product(range(n_steps + 1), repeat=len(condition_names)):
        if sum(counts) != n_steps:
            continue
        weights.append(
            {
                condition: float(count * step)
                for condition, count in zip(condition_names, counts, strict=True)
            }
        )
    return weights


def parse_grid_specs(specs: list[str]) -> list[tuple[list[str], float]]:
    parsed = []
    for spec in specs:
        names_part, step_part = spec.split(":", 1)
        names = [name for name in names_part.split(",") if name]
        if not names:
            raise ValueError(f"No condition names in grid spec: {spec}")
        parsed.append((names, float(step_part)))
    return parsed


def candidate_weights(conditions: list[str], grid_specs: list[tuple[list[str], float]]) -> list[dict[str, float]]:
    candidates = [{condition: 1.0} for condition in conditions]
    for names, step in grid_specs:
        missing = [name for name in names if name not in conditions]
        if missing:
            raise ValueError(f"Unknown condition(s) in grid: {missing}")
        for local in convex_grid(names, step):
            weights = {condition: 0.0 for condition in conditions}
            weights.update(local)
            candidates.append(weights)

    deduped = []
    seen = set()
    for weights in candidates:
        key = tuple(round(weights.get(condition, 0.0), 10) for condition in conditions)
        if key not in seen:
            seen.add(key)
            deduped.append(weights)
    return deduped


def fuse_predictions(preds: np.ndarray, conditions: list[str], weights: dict[str, float]) -> np.ndarray:
    coeff = np.array([weights.get(condition, 0.0) for condition in conditions], dtype=np.float32)
    return np.einsum("rcu,c->ru", preds, coeff, optimize=True)


def score_weights(
    preds: np.ndarray,
    target: np.ndarray,
    conditions: list[str],
    weights: dict[str, float],
) -> dict[str, Any]:
    fused = fuse_predictions(preds, conditions, weights)
    metrics = vertexwise_pearson(fused, target)
    return {
        "weights": {condition: float(weights.get(condition, 0.0)) for condition in conditions},
        "mean": metrics["mean"],
        "median": metrics["median"],
        "n_vertices_finite": metrics["n_vertices_finite"],
    }


def select_best(scores: list[dict[str, Any]], full_condition: str = "full") -> dict[str, Any]:
    finite = [score for score in scores if score["mean"] is not None and math.isfinite(score["mean"])]
    if not finite:
        raise ValueError("No finite fusion scores")

    def key(score: dict[str, Any]) -> tuple[float, float]:
        full_weight = float(score["weights"].get(full_condition, 0.0))
        return float(score["mean"]), full_weight

    return max(finite, key=key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-condition", nargs="+", action="append", required=True)
    parser.add_argument("--val-condition", nargs="+", action="append", required=True)
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Convex grid spec like full,text_video:0.05",
    )
    parser.add_argument("--full-condition", default="full")
    parser.add_argument("--time-tolerance-seconds", type=float, default=0.25)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/ablation_results/tribe_late_fusion_ablation.json"),
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

    grids = parse_grid_specs(args.grid)
    candidates = candidate_weights(conditions, grids)
    train_scores = [
        score_weights(train_preds, train_target, conditions, weights)
        for weights in candidates
    ]
    best_train = select_best(train_scores, full_condition=args.full_condition)
    val_selected = score_weights(
        val_preds, val_target, conditions, best_train["weights"]
    )
    full_weights = {condition: 0.0 for condition in conditions}
    full_weights[args.full_condition] = 1.0
    val_full = score_weights(val_preds, val_target, conditions, full_weights)
    delta_val = (
        float(val_selected["mean"] - val_full["mean"])
        if val_selected["mean"] is not None and val_full["mean"] is not None
        else None
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "tribe_late_fusion_ablation",
        "conditions": conditions,
        "grids": args.grid,
        "time_tolerance_seconds": args.time_tolerance_seconds,
        "n_candidates": len(candidates),
        "train_shape": list(train_preds.shape),
        "val_shape": list(val_preds.shape),
        "train_condition_paths": {
            condition: [str(path) for path in paths]
            for condition, paths in train_paths.items()
        },
        "val_condition_paths": {
            condition: [str(path) for path in paths]
            for condition, paths in val_paths.items()
        },
        "best_train": best_train,
        "val_selected": val_selected,
        "val_full": val_full,
        "delta_val_selected_vs_full": delta_val,
        "runtime_seconds": time.perf_counter() - start,
        "caveat": (
            "Available-label local proxy only. Weights are global convex "
            "late-fusion coefficients selected on train chunks, not hidden "
            "Friends S7/OOD labels."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    print(args.out)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
