import torch

from tribev2.losses import PearsonCorrelationLoss, PearsonMSELoss


def test_pearson_correlation_loss_prefers_correlated_predictions():
    y_true = torch.randn(16, 8)
    good_pred = y_true * 2.0 + 3.0
    bad_pred = -y_true

    loss = PearsonCorrelationLoss(dim=0, reduction="mean").build()

    assert loss(good_pred, y_true) < 1e-5
    assert loss(bad_pred, y_true) > 1.9


def test_pearson_mse_loss_returns_per_output_vector():
    y_true = torch.randn(12, 5)
    y_pred = y_true + 0.1 * torch.randn(12, 5)

    loss = PearsonMSELoss(dim=0, reduction="none").build()
    out = loss(y_pred, y_true)

    assert out.shape == (5,)
    assert torch.isfinite(out).all()
