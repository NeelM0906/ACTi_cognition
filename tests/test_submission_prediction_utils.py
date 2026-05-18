from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.predict_tribe_algonauts_submission import (
    discover_items,
    fill_nonfinite_predictions,
    get_surface_prediction_for_item,
    resample_predictions_to_target_samples,
    stimulus_paths,
)


def test_resample_predictions_to_target_samples_preserves_shape_and_edges():
    preds = np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32)
    sample_times = np.array([1.0, 3.0], dtype=np.float32)

    out = resample_predictions_to_target_samples(
        pred_parcels=preds,
        sample_times=sample_times,
        n_samples=4,
        native_frequency=1.0,
    )

    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[0], [1.0, 10.0])
    np.testing.assert_allclose(out[2], [2.0, 20.0])
    np.testing.assert_allclose(out[3], [3.0, 30.0])


def test_resample_predictions_to_target_samples_handles_duplicate_times():
    preds = np.array([[1.0], [99.0], [3.0]], dtype=np.float32)
    sample_times = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    out = resample_predictions_to_target_samples(
        pred_parcels=preds,
        sample_times=sample_times,
        n_samples=3,
        native_frequency=1.0,
    )

    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.0])


def test_fill_nonfinite_predictions_with_row_mean():
    preds = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]], dtype=np.float32)

    out, n_filled = fill_nonfinite_predictions(preds, method="row-mean")

    assert n_filled == 4
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 0.0])


def test_fill_nonfinite_predictions_with_zero():
    preds = np.array([[1.0, np.inf]], dtype=np.float32)

    out, n_filled = fill_nonfinite_predictions(preds, method="zero")

    assert n_filled == 1
    np.testing.assert_allclose(out, [[1.0, 0.0]])


def test_discover_items_filters_requested_items(tmp_path):
    sample_dir = tmp_path / "download" / "algonauts_2025.competitors" / "fmri"
    sample_dir = sample_dir / "sub-01" / "target_sample_number"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "sub-01_friends-s7_fmri_samples.npy", {"s07e01a": 2, "s07e01b": 3})

    items = discover_items(
        tmp_path,
        "sub-01",
        "friends-s7",
        requested_items=["s07e01b"],
    )

    assert items == {"s07e01b": 3}


def test_stimulus_paths_for_friends_and_ood():
    root = Path("/data")

    friends_movie, friends_transcript = stimulus_paths(root, "friends-s7", "s07e01a")
    ood_movie, ood_transcript = stimulus_paths(root, "ood", "pulpfiction1")
    chaplin_movie, chaplin_transcript = stimulus_paths(root, "ood", "chaplin1")

    assert str(friends_movie).endswith("stimuli/movies/friends/s7/friends_s07e01a.mkv")
    assert str(friends_transcript).endswith(
        "stimuli/transcripts/friends/s7/friends_s07e01a.tsv"
    )
    assert str(ood_movie).endswith("stimuli/movies/ood/pulpfiction/task-pulpfiction1_video.mkv")
    assert str(ood_transcript).endswith(
        "stimuli/transcripts/ood/pulpfiction/ood_pulpfiction1.tsv"
    )
    assert str(chaplin_movie).endswith("stimuli/movies/ood/chaplin/task-chaplin1_video.mkv")
    assert chaplin_transcript is None


def test_surface_prediction_for_item_reuses_memory_cache(monkeypatch, tmp_path):
    calls = {"events": 0, "predict": 0}

    def fake_load_submission_events(*args, **kwargs):
        calls["events"] += 1
        return pd.DataFrame({"type": ["Video"], "start": [0.0]})

    class FakeModel:
        def predict(self, events, verbose=False):
            calls["predict"] += 1
            return (
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                [SimpleNamespace(start=0.0), SimpleNamespace(start=1.0)],
            )

    monkeypatch.setattr(
        "experiments.predict_tribe_algonauts_submission.load_submission_events",
        fake_load_submission_events,
    )
    stimulus_cache = {}

    first = get_surface_prediction_for_item(
        model=FakeModel(),
        data_path=tmp_path,
        checkpoint="facebook/tribev2",
        phase="friends-s7",
        subject="sub-01",
        item="s07e01a",
        features=["text", "audio", "video"],
        target_offset_seconds=5.0,
        prediction_cache_dir=None,
        stimulus_cache=stimulus_cache,
    )
    second = get_surface_prediction_for_item(
        model=FakeModel(),
        data_path=tmp_path,
        checkpoint="facebook/tribev2",
        phase="friends-s7",
        subject="sub-02",
        item="s07e01a",
        features=["text", "audio", "video"],
        target_offset_seconds=5.0,
        prediction_cache_dir=None,
        stimulus_cache=stimulus_cache,
    )

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], [5.0, 6.0])
    assert first[2] == 2
    assert first[3] == "computed"
    assert second[3] == "memory"
    assert calls == {"events": 1, "predict": 1}
