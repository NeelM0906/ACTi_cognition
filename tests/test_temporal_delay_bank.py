import torch

from tribev2.model import FmriEncoder, TemporalDelayBank


def test_temporal_delay_bank_identity_initialization_preserves_input():
    x = torch.randn(2, 7, 4)
    module = TemporalDelayBank(n_lags=3, lag_stride=2).build(dim=4)

    out = module(x)

    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_temporal_delay_bank_can_shift_past_features():
    x = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    module = TemporalDelayBank(n_lags=2, lag_stride=1).build(dim=1)
    with torch.no_grad():
        module.weights.zero_()
        module.weights[:, 1] = 1.0

    out = module(x)

    expected = torch.tensor([[[0.0], [0.0], [1.0], [2.0], [3.0]]])
    assert torch.equal(out, expected)


def test_fmri_encoder_with_delay_bank_preserves_output_shape():
    config = FmriEncoder(
        hidden=64,
        low_rank_head=8,
        encoder={"depth": 1, "heads": 2},
        temporal_delay_bank={
            "name": "TemporalDelayBank",
            "n_lags": 3,
            "lag_stride": 1,
        },
    )
    model = config.build(
        feature_dims={"text": (1, 6)},
        n_outputs=5,
        n_output_timesteps=4,
    )

    class Batch:
        data = {
            "text": torch.randn(2, 1, 6, 10),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }

    out = model(Batch())

    assert out.shape == (2, 5, 4)


def test_fmri_encoder_with_output_delay_bank_preserves_output_shape():
    config = FmriEncoder(
        hidden=64,
        low_rank_head=8,
        encoder={"depth": 1, "heads": 2},
        output_temporal_delay_bank={
            "name": "TemporalDelayBank",
            "n_lags": 3,
            "lag_stride": 1,
        },
    )
    model = config.build(
        feature_dims={"text": (1, 6)},
        n_outputs=5,
        n_output_timesteps=4,
    )

    class Batch:
        data = {
            "text": torch.randn(2, 1, 6, 10),
            "subject_id": torch.zeros(2, dtype=torch.long),
        }

    out = model(Batch())

    assert out.shape == (2, 5, 4)
