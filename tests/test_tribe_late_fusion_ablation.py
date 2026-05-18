import numpy as np
import pytest

from experiments.tribe_late_fusion_ablation import (
    ConditionRun,
    common_time_indices,
    convex_grid,
    fuse_predictions,
    select_best,
)


def test_convex_grid_generates_weights_that_sum_to_one():
    weights = convex_grid(["a", "b"], step=0.5)

    assert weights == [
        {"a": 0.0, "b": 1.0},
        {"a": 0.5, "b": 0.5},
        {"a": 1.0, "b": 0.0},
    ]


def test_common_time_indices_aligns_overlapping_times():
    a = ConditionRun(
        condition="a",
        result_path=None,
        preds=np.zeros((3, 1), dtype=np.float32),
        target=np.zeros((3, 1), dtype=np.float32),
        sample_times=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    b = ConditionRun(
        condition="b",
        result_path=None,
        preds=np.zeros((3, 1), dtype=np.float32),
        target=np.zeros((3, 1), dtype=np.float32),
        sample_times=np.array([2.0, 3.0, 4.0], dtype=np.float32),
    )

    indices = common_time_indices([a, b])

    assert indices["a"].tolist() == [1, 2]
    assert indices["b"].tolist() == [0, 1]


def test_common_time_indices_allows_small_time_tolerance():
    a = ConditionRun(
        condition="a",
        result_path=None,
        preds=np.zeros((2, 1), dtype=np.float32),
        target=np.zeros((2, 1), dtype=np.float32),
        sample_times=np.array([23.97, 24.97], dtype=np.float32),
    )
    b = ConditionRun(
        condition="b",
        result_path=None,
        preds=np.zeros((2, 1), dtype=np.float32),
        target=np.zeros((2, 1), dtype=np.float32),
        sample_times=np.array([24.0, 25.0], dtype=np.float32),
    )

    indices = common_time_indices([a, b], tolerance_seconds=0.06)

    assert indices["a"].tolist() == [0, 1]
    assert indices["b"].tolist() == [0, 1]


def test_fuse_predictions_applies_condition_weights():
    preds = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )

    fused = fuse_predictions(preds, ["a", "b"], {"a": 0.25, "b": 0.75})

    np.testing.assert_allclose(fused, [[4.0, 6.0], [5.0, 7.0]])


def test_select_best_tie_breaks_toward_full_weight():
    scores = [
        {"mean": 0.2, "weights": {"full": 0.0}},
        {"mean": 0.2, "weights": {"full": 1.0}},
        {"mean": 0.1, "weights": {"full": 0.0}},
    ]

    assert select_best(scores)["weights"]["full"] == pytest.approx(1.0)
