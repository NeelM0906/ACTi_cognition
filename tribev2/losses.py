# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Losses for fMRI response prediction."""

import typing as tp

import torch
from torch import nn

from neuraltrain.losses import BaseLoss


def _reduce_loss(
    loss: torch.Tensor, reduction: tp.Literal["none", "mean", "sum"]
) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


class PearsonCorrelationLossImpl(nn.Module):
    """Differentiable 1 - Pearson r loss along one dimension."""

    def __init__(
        self,
        dim: int = 0,
        eps: float = 1e-8,
        reduction: tp.Literal["none", "mean", "sum"] = "none",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pred = y_pred - y_pred.mean(dim=self.dim, keepdim=True)
        true = y_true - y_true.mean(dim=self.dim, keepdim=True)
        numerator = (pred * true).sum(dim=self.dim)
        denominator = pred.square().sum(dim=self.dim).sqrt()
        denominator = denominator * true.square().sum(dim=self.dim).sqrt()
        corr = numerator / denominator.clamp_min(self.eps)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return _reduce_loss(1.0 - corr, self.reduction)


class PearsonMSELossImpl(nn.Module):
    """Per-output Pearson loss with a stabilizing MSE term."""

    def __init__(
        self,
        pearson_weight: float = 1.0,
        mse_weight: float = 0.05,
        dim: int = 0,
        eps: float = 1e-8,
        reduction: tp.Literal["none", "mean", "sum"] = "none",
    ) -> None:
        super().__init__()
        self.pearson = PearsonCorrelationLossImpl(
            dim=dim,
            eps=eps,
            reduction="none",
        )
        self.pearson_weight = pearson_weight
        self.mse_weight = mse_weight
        self.dim = dim
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pearson_loss = self.pearson(y_pred, y_true)
        mse_loss = (y_pred - y_true).square().mean(dim=self.dim)
        loss = self.pearson_weight * pearson_loss + self.mse_weight * mse_loss
        return _reduce_loss(loss, self.reduction)


class PearsonCorrelationLoss(BaseLoss):
    dim: int = 0
    eps: float = 1e-8
    reduction: tp.Literal["none", "mean", "sum"] = "none"

    def build(self, **kwargs: tp.Any) -> nn.Module:
        if kwargs:
            raise ValueError(f"Unexpected build kwargs: {sorted(kwargs)}")
        return PearsonCorrelationLossImpl(
            dim=self.dim,
            eps=self.eps,
            reduction=self.reduction,
        )


class PearsonMSELoss(BaseLoss):
    pearson_weight: float = 1.0
    mse_weight: float = 0.05
    dim: int = 0
    eps: float = 1e-8
    reduction: tp.Literal["none", "mean", "sum"] = "none"

    def build(self, **kwargs: tp.Any) -> nn.Module:
        if kwargs:
            raise ValueError(f"Unexpected build kwargs: {sorted(kwargs)}")
        return PearsonMSELossImpl(
            pearson_weight=self.pearson_weight,
            mse_weight=self.mse_weight,
            dim=self.dim,
            eps=self.eps,
            reduction=self.reduction,
        )
