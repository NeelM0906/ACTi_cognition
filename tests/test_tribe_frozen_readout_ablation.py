import numpy as np

from experiments.tribe_frozen_readout_ablation import (
    episode_identity,
    finite_unit_mask,
    fit_ridge_readout,
    parse_run_spec,
    predict_ridge,
    select_alpha,
    validate_disjoint_runs,
    validate_feature_alignment,
)


def _payload(preds, target):
    preds = np.asarray(preds, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    rows = np.ones(preds.shape[0], dtype=bool)
    return {
        "preds": preds,
        "target": target,
        "sample_times": np.arange(preds.shape[0], dtype=np.float32),
        "finite_rows": rows,
    }


def test_parse_run_spec_supports_paths_with_colons_after_third_field():
    spec = parse_run_spec("e01a=sub-01:s01:e01a:/tmp/a:b/file.npz")

    assert spec.label == "e01a"
    assert spec.subject == "sub-01"
    assert spec.movie == "s01"
    assert spec.chunk == "e01a"
    assert str(spec.baseline_npz) == "/tmp/a:b/file.npz"


def test_finite_unit_mask_uses_common_baseline_finite_units():
    run = _payload(
        preds=[[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]],
        target=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    )

    mask = finite_unit_mask([run])

    np.testing.assert_array_equal(mask, [True, False, True])


def test_validate_disjoint_runs_rejects_same_underlying_chunk():
    train = [parse_run_spec("train=sub-01:s01:e01a:/tmp/a.npz")]
    val = [parse_run_spec("val=sub-01:s01:e01a:/tmp/b.npz")]

    try:
        validate_disjoint_runs(train, val)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected overlap to be rejected")


def test_validate_disjoint_runs_rejects_same_episode_halves():
    train = [parse_run_spec("train=sub-01:s01:e01a:/tmp/a.npz")]
    val = [parse_run_spec("val=sub-01:s01:e01b:/tmp/b.npz")]

    try:
        validate_disjoint_runs(train, val)
    except ValueError as exc:
        assert "episode overlap" in str(exc)
    else:
        raise AssertionError("Expected same-episode halves to be rejected")


def test_episode_identity_strips_chunk_half_suffix():
    spec = parse_run_spec("run=sub-01:s01:e12b:/tmp/a.npz")

    assert episode_identity(spec) == ("sub-01", "s01", "e12")


def test_validate_feature_alignment_rejects_shifted_rows():
    spec = parse_run_spec("e01a=sub-01:s01:e01a:/tmp/a.npz")
    feature_payload = {
        "segment_starts": np.array([0.0, 1.0, 2.0], dtype=np.float32)
    }
    baseline_payload = {
        "sample_times": np.array([5.0, 6.0, 7.2], dtype=np.float32)
    }

    try:
        validate_feature_alignment(
            spec,
            feature_payload,
            baseline_payload,
            target_offset_seconds=5.0,
        )
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("Expected row alignment mismatch to be rejected")


def test_fit_ridge_readout_recovers_linear_mapping():
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.concatenate([2 * x + 1, -x + 4], axis=1)

    fitted = fit_ridge_readout(x, y, alpha=1e-6)
    pred = predict_ridge(x, fitted)

    np.testing.assert_allclose(pred, y, atol=1e-4)


def test_select_alpha_reports_inner_train_folds():
    specs = [
        parse_run_spec("a=sub-01:s01:e01a:/tmp/a.npz"),
        parse_run_spec("b=sub-01:s01:e01b:/tmp/b.npz"),
    ]
    feature_payloads = {
        "a": {"features": np.array([[0.0], [1.0]], dtype=np.float32)},
        "b": {"features": np.array([[2.0], [3.0]], dtype=np.float32)},
    }
    baseline_payloads = {
        "a": _payload(preds=[[0.0], [0.0]], target=[[0.0], [1.0]]),
        "b": _payload(preds=[[0.0], [0.0]], target=[[2.0], [3.0]]),
    }

    result = select_alpha(
        specs,
        feature_payloads,
        baseline_payloads,
        unit_mask=np.array([True]),
        alphas=[1e-6, 1000.0],
    )

    assert result["selected_alpha"] in {1e-06, 1000.0}
    assert len(result["inner_scores"]) == 2
    assert all(len(row["fold_scores"]) == 2 for row in result["inner_scores"])
