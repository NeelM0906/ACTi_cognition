import numpy as np

from experiments.tribe_calibration_ablation import (
    _fit_global_ridge,
    _lagged_design,
    _split_rows,
    _valid_rows_for_lags,
)


def test_split_rows_leaves_gap_between_train_and_validation():
    train, val = _split_rows(n_rows=10, train_frac=0.5, gap=2)

    np.testing.assert_array_equal(train, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(val, [7, 8, 9])


def test_lagged_design_uses_negative_lag_for_future_rows():
    series = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    design = _lagged_design(series, lags=[-1, 0, 1])

    np.testing.assert_allclose(design[:, 0, 0], [2.0, 3.0, 4.0, np.nan])
    np.testing.assert_allclose(design[:, 0, 1], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(design[:, 0, 2], [np.nan, 1.0, 2.0, 3.0])


def test_valid_rows_for_lags_trims_edges():
    rows = np.arange(10)

    valid = _valid_rows_for_lags(n_rows=10, rows=rows, lags=[-2, 0, 3])

    np.testing.assert_array_equal(valid, [3, 4, 5, 6, 7])


def test_global_ridge_can_use_cross_parcel_signal():
    preds = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [6.0, -1.0],
        ],
        dtype=np.float32,
    )
    target = np.column_stack([preds[:, 1] * 2.0, preds[:, 0] * -1.0]).astype(
        np.float32
    )

    calibrated = _fit_global_ridge(
        preds=preds,
        target=target,
        train_rows=np.arange(4),
        val_rows=np.arange(4, 6),
        lags=[0],
        alpha=1e-6,
    )

    np.testing.assert_allclose(calibrated[4:6], target[4:6], atol=1e-4)


def test_global_ridge_keeps_baseline_invalid_units_unscored():
    preds = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
            [4.0, np.nan],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [2.0, 10.0],
            [4.0, 11.0],
            [6.0, 12.0],
            [8.0, 13.0],
        ],
        dtype=np.float32,
    )

    calibrated = _fit_global_ridge(
        preds=preds,
        target=target,
        train_rows=np.arange(2),
        val_rows=np.arange(2, 4),
        lags=[0],
        alpha=1e-6,
    )

    assert np.isfinite(calibrated[2:4, 0]).all()
    assert np.isnan(calibrated[2:4, 1]).all()
