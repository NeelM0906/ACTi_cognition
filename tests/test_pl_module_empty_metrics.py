import torch
from torch import nn
from torchmetrics import PearsonCorrCoef

from tribev2.pl_module import BrainModule


class _Batch:
    def __init__(self, fmri: torch.Tensor) -> None:
        self.data = {
            "fmri": fmri,
            "subject_id": torch.tensor([0]),
        }


class _Model(nn.Module):
    def forward(self, batch: _Batch) -> torch.Tensor:
        return torch.randn_like(batch.data["fmri"], requires_grad=True)


def test_validation_step_skips_empty_metric_update() -> None:
    module = BrainModule(
        model=_Model(),
        loss=nn.MSELoss(reduction="none"),
        optim_config=None,
        metrics={"val/pearson": PearsonCorrCoef(num_outputs=3)},
        config={
            "data.overlap_trs_val": 0,
            "data.stride_drop_incomplete": False,
            "max_steps": 0,
        },
    )
    loss, y_pred, y_true = module._run_step(
        _Batch(torch.zeros(1, 3, 4)), 0, step_name="val"
    )

    assert loss.item() == 0.0
    assert y_pred.shape == (1, 3, 4)
    assert y_true.shape == (1, 3, 4)


def test_validation_step_filters_non_finite_targets() -> None:
    fmri = torch.randn(1, 3, 4)
    fmri[:, :, -1] = float("nan")
    module = BrainModule(
        model=_Model(),
        loss=nn.MSELoss(reduction="none"),
        optim_config=None,
        metrics={"val/pearson": PearsonCorrCoef(num_outputs=3)},
        config={
            "data.overlap_trs_val": 0,
            "data.stride_drop_incomplete": False,
            "max_steps": 0,
        },
    )

    loss, _, _ = module._run_step(_Batch(fmri), 0, step_name="val")

    assert torch.isfinite(loss)
