import numpy as np

from experiments.calibrate_tribe_submission import (
    apply_calibration_to_prediction,
    select_alpha_by_episode,
)
from experiments.tribe_frozen_readout_ablation import RunSpec


def test_apply_calibration_preserves_uncalibrated_units():
    raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    unit_mask = np.array([True, False, True])
    fitted = {
        "weights": np.eye(2, dtype=np.float32),
        "x_mean": np.zeros((1, 2), dtype=np.float32),
        "y_mean": np.ones((1, 2), dtype=np.float32),
    }

    calibrated = apply_calibration_to_prediction(raw, fitted, unit_mask)

    np.testing.assert_allclose(calibrated[:, [0, 2]], raw[:, [0, 2]] + 1.0)
    np.testing.assert_allclose(calibrated[:, 1], raw[:, 1])


def test_select_alpha_by_episode_holds_out_complete_episode_groups():
    specs = [
        RunSpec("e01a", "sub-01", "s01", "e01a", "unused"),
        RunSpec("e01b", "sub-01", "s01", "e01b", "unused"),
        RunSpec("e02a", "sub-01", "s01", "e02a", "unused"),
        RunSpec("e02b", "sub-01", "s01", "e02b", "unused"),
    ]
    unit_mask = np.array([True])
    feature_payloads = {}
    baseline_payloads = {}
    for index, spec in enumerate(specs):
        value = np.array([[float(index)]], dtype=np.float32)
        feature_payloads[spec.label] = {"features": value}
        baseline_payloads[spec.label] = {
            "preds": value,
            "target": value,
            "sample_times": np.array([float(index)], dtype=np.float32),
            "finite_rows": np.array([True]),
        }

    result = select_alpha_by_episode(
        specs,
        feature_payloads,
        baseline_payloads,
        unit_mask,
        [1.0, 10.0],
    )

    fold_groups = [
        {fold["episode"] for fold in score["fold_scores"]}
        for score in result["inner_scores"]
    ]
    assert fold_groups == [
        {"sub-01/s01/e01", "sub-01/s01/e02"},
        {"sub-01/s01/e01", "sub-01/s01/e02"},
    ]
