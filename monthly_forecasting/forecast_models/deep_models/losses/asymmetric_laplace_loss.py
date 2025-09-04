import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AsymmetricLaplaceLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu, beta, tau, y_true):
        """
        Args:
            mu: (B,)  predicted mean (location parameter)
            b: (B,)   predicted scale (>0)
            tau: (B,) predicted asymmetry (0-1)
            y_true: (B,) ground truth
        """

        # 2) Compute piecewise ρτ(y−μ)
        diff = y_true - mu
        rho = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)

        # 3) Assemble per-sample NLL
        nll = torch.log(beta) - torch.log(tau) - torch.log(1 - tau) + rho / beta

        # 4) Reduction
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll
