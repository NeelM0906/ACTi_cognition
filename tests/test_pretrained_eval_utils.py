import numpy as np
import pandas as pd
import pytest

from experiments.eval_pretrained_tribe_algonauts import (
    prune_unrequested_events,
    resample_parcels_to_segments,
    surface_to_parcels,
    vertexwise_pearson,
)


def test_surface_to_parcels_averages_projected_vertices():
    surface = np.array(
        [
            [1.0, 3.0, 10.0, 20.0, 99.0],
            [5.0, 7.0, 30.0, 40.0, 99.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 1, 2, 2, 0], dtype=np.int32)

    parcels = surface_to_parcels(surface, labels, n_parcels=3)

    np.testing.assert_allclose(parcels[:, 0], [2.0, 6.0])
    np.testing.assert_allclose(parcels[:, 1], [15.0, 35.0])
    assert np.isnan(parcels[:, 2]).all()


def test_resample_parcels_to_segments_interpolates_time_axis():
    parcel_data = np.array(
        [
            [0.0, 10.0],
            [2.0, 12.0],
            [4.0, 14.0],
        ],
        dtype=np.float32,
    )

    resampled = resample_parcels_to_segments(
        parcel_data, native_frequency=1.0, sample_times=np.array([0.5, 1.5])
    )

    np.testing.assert_allclose(resampled, [[1.0, 11.0], [3.0, 13.0]])


def test_vertexwise_pearson_reports_finite_units():
    pred = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    target = np.array([[2.0, 4.0], [4.0, 4.0], [6.0, 4.0]], dtype=np.float32)

    metrics = vertexwise_pearson(pred, target)

    assert metrics["mean"] == pytest.approx(1.0)
    assert metrics["median"] == pytest.approx(1.0)
    assert metrics["n_vertices_finite"] == 1


def test_prune_unrequested_events_keeps_video_as_audio_source_before_transforms():
    events = pd.DataFrame(
        {
            "type": ["Video", "Audio", "Word", "Text"],
            "start": [0.0, 0.0, 0.5, 0.5],
        }
    )

    before = prune_unrequested_events(events, ["audio"], before_transforms=True)
    assert before["type"].tolist() == ["Video", "Audio"]

    after_audio = pd.DataFrame({"type": ["Video", "Audio"], "start": [0.0, 0.0]})
    after = prune_unrequested_events(after_audio, ["audio"], before_transforms=False)
    assert after["type"].tolist() == ["Audio"]
