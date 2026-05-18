import numpy as np

from experiments.tribe_native_tr_subject_adaptation import (
    native_lowrank_features,
    resample_to_native_grid,
)
from experiments.tribe_frozen_readout_ablation import parse_run_spec


def test_resample_to_native_grid_edge_fills_and_interpolates_duplicates():
    values = np.array([[1.0], [99.0], [5.0]], dtype=np.float32)
    sample_times = np.array([1.0, 1.0, 3.0], dtype=np.float32)

    out = resample_to_native_grid(
        values,
        sample_times,
        n_samples=5,
        native_frequency=1.0,
    )

    np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 3.0, 5.0, 5.0])


def test_native_lowrank_features_uses_shifted_segment_times():
    spec = parse_run_spec("e01a=sub-01:s01:e01a:/tmp/baseline.npz")
    feature_payload = {
        "features": np.array([[10.0], [20.0]], dtype=np.float32),
        "segment_starts": np.array([0.0, 2.0], dtype=np.float32),
    }
    native_payload = {
        "target": np.zeros((4, 1), dtype=np.float32),
    }

    out = native_lowrank_features(
        spec,
        feature_payload,
        native_payload,
        native_frequency=1.0,
        target_offset_seconds=1.0,
    )

    np.testing.assert_allclose(out["features"][:, 0], [10.0, 10.0, 15.0, 20.0])
