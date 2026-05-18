import numpy as np
import pytest

from experiments.tribe_regionwise_fusion_ablation import (
    apply_regionwise_weights,
    columnwise_pearson,
    fit_regionwise_weights,
)


def test_columnwise_pearson_scores_each_unit():
    pred = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    target = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

    corr = columnwise_pearson(pred, target)

    assert corr[0] == pytest.approx(1.0)
    assert corr[1] == pytest.approx(-1.0)


def test_fit_regionwise_weights_selects_different_units():
    train_preds = np.array(
        [
            [[1.0, 3.0], [3.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 1.0], [1.0, 3.0]],
        ],
        dtype=np.float32,
    )
    target = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    candidates = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]

    weights, selected_corr = fit_regionwise_weights(
        train_preds,
        target,
        conditions=["a", "b"],
        candidates=candidates,
    )

    np.testing.assert_allclose(weights, [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(selected_corr, [1.0, 1.0])


def test_apply_regionwise_weights_fuses_per_unit():
    preds = np.array(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[2.0, 20.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )
    weights = np.array([[1.0, 0.0], [0.25, 0.75]], dtype=np.float32)

    fused = apply_regionwise_weights(preds, weights)

    np.testing.assert_allclose(fused, [[1.0, 25.0], [2.0, 35.0]])
