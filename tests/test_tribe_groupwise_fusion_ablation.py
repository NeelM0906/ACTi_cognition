import numpy as np

from experiments.tribe_groupwise_fusion_ablation import (
    apply_groupwise_weights,
    fit_groupwise_weights,
)


def test_fit_groupwise_weights_shares_choice_within_group():
    train_preds = np.array(
        [
            [[1.0, 1.0, 3.0], [3.0, 3.0, 1.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 1.0], [1.0, 1.0, 3.0]],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    groups = np.array([0, 0, 1], dtype=np.int64)
    candidates = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]

    weights = fit_groupwise_weights(
        train_preds,
        target,
        conditions=["a", "b"],
        candidates=candidates,
        groups=groups,
    )

    np.testing.assert_allclose(weights[:2], [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(weights[2], [0.0, 1.0])


def test_apply_groupwise_weights_uses_unit_weights():
    preds = np.array(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[2.0, 20.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )
    weights = np.array([[1.0, 0.0], [0.25, 0.75]], dtype=np.float32)

    fused = apply_groupwise_weights(preds, weights)

    np.testing.assert_allclose(fused, [[1.0, 25.0], [2.0, 35.0]])
