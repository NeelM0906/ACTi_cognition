from pathlib import Path

import numpy as np
import pytest

from experiments.tribe_hrf_offset_ablation import (
    PredictionRun,
    score_runs_at_offset,
    select_best_offset,
    targets_for_offset,
)


def _run(segment_starts, preds):
    return PredictionRun(
        result_path=Path("result.json"),
        arrays_path=Path("arrays.npz"),
        subject="sub-01",
        movie="s01",
        chunks=("e01a",),
        baseline_offset_seconds=5.0,
        preds=np.asarray(preds, dtype=np.float32),
        segment_starts=np.asarray(segment_starts, dtype=np.float32),
    )


def test_targets_for_offset_resamples_from_segment_starts():
    run = _run(segment_starts=[0.0, 1.0], preds=[[0.0], [0.0]])

    def loader(_data_path, _subject, _movie, _chunk):
        return np.array([[10.0], [20.0], [30.0]], dtype=np.float32)

    target, finite_rows = targets_for_offset(
        run,
        data_path=Path("/tmp/data"),
        offset_seconds=1.0,
        native_frequency=1.0,
        parcel_loader=loader,
    )

    np.testing.assert_allclose(target[:, 0], [20.0, 30.0])
    assert finite_rows.tolist() == [True, True]


def test_score_runs_at_offset_prefers_aligned_offset():
    run = _run(segment_starts=[0.0, 1.0, 2.0], preds=[[1.0], [0.0], [1.0]])

    def loader(_data_path, _subject, _movie, _chunk):
        return np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)

    aligned = score_runs_at_offset(
        [run],
        data_path=Path("/tmp/data"),
        offset_seconds=1.0,
        native_frequency=1.0,
        parcel_loader=loader,
    )
    shifted = score_runs_at_offset(
        [run],
        data_path=Path("/tmp/data"),
        offset_seconds=0.0,
        native_frequency=1.0,
        parcel_loader=loader,
    )

    assert aligned["mean"] == pytest.approx(1.0)
    assert shifted["mean"] == pytest.approx(-1.0)
    assert aligned["n_finite_rows"] == 3


def test_select_best_offset_tie_breaks_toward_default_five_seconds():
    rows = [
        {"offset_seconds": 4.0, "mean": 0.2},
        {"offset_seconds": 5.0, "mean": 0.2},
        {"offset_seconds": 6.0, "mean": 0.19},
    ]

    assert select_best_offset(rows)["offset_seconds"] == 5.0
