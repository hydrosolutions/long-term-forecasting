import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import logging

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)



def predict_ALD_quantile(mu, b, tau, q, eps=1e-12):
        """
        Quantile function for Asymmetric Laplace in (m=mu, lambda=b, p=tau) parametrization.

        mu  : array-like, location
        b   : array-like/float, scale (>0)
        tau : array-like/float in (0,1), asymmetry (quantile pivot)
        q   : scalar or array-like in (0,1), desired quantile(s)
        """
        mu = np.asarray(mu, dtype=float)
        b = np.asarray(b, dtype=float)
        tau = np.asarray(tau, dtype=float)

        # Clamp parameters for numerical safety
        tau = np.clip(tau, eps, 1.0 - eps)
        b = np.maximum(b, eps)

        # Broadcast q to mu's shape
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape == ():  # scalar q
            q_arr = np.full_like(mu, np.clip(q_arr, eps, 1 - eps))
        else:
            q_arr = np.clip(q_arr, eps, 1 - eps)
            q_arr = np.broadcast_to(q_arr, mu.shape)

        # Scale = b / (tau*(1-tau))  (your earlier choice)
        scale = b / (tau * (1.0 - tau))

        # Piecewise branches with broadcasting, no indexing of q
        left = mu + scale * tau * np.log(np.clip(q_arr / tau, eps, None))
        right = mu - scale * (1.0 - tau) * np.log(
            np.clip((1.0 - q_arr) / (1.0 - tau), eps, None)
        )

        return np.where(q_arr <= tau, left, right)

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
