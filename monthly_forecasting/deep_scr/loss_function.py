import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


    
class AsymmetricLaplaceLoss(nn.Module):
    def __init__(self, reduction='mean'):
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
        rho = torch.where(
            diff >= 0,
            tau * diff,
            (tau - 1) * diff
        )

        # 3) Assemble per-sample NLL
        nll = (
            torch.log(beta)
            - torch.log(tau)
            - torch.log(1 - tau)
            + rho / beta
        )

        # 4) Reduction
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll

class QuantileLoss(nn.Module):
    """
    Quantile loss (pinball) with optional soft center penalty (only penalize large drifts).
    """
    def __init__(self, 
                 quantiles: List[float], 
                 reduction: str = 'mean',
                 center_weight: float = 0.0,
                 center_eps: float = 0.1):
        """
        Args:
            quantiles (List[float]): Quantiles between 0 and 1.
            reduction (str): 'none', 'mean', or 'sum'.
            center_weight (float): Weight of the center penalty.
            center_eps (float): Slack margin for center penalty (default 0.0 = always penalize).
        """
        super().__init__()
        assert all(0 < q < 1 for q in quantiles), "Quantiles must be in (0,1)"
        assert reduction in ['none', 'mean', 'sum']
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))
        self.reduction = reduction
        self.center_weight = center_weight
        self.center_eps = center_eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor, center_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure correct shapes
        if target.dim() == 1:
            target = target.unsqueeze(1)

        q = self.quantiles.unsqueeze(0)  # (1, num_quantiles)
        errors = target - preds

        pinball_loss = torch.max(q * errors, (q - 1) * errors)

        if self.reduction == 'mean':
            loss = pinball_loss.mean()
        elif self.reduction == 'sum':
            loss = pinball_loss.sum()
        else:  # 'none'
            loss = pinball_loss  # (B, num_quantiles)

        # Optional center penalty
        if center_target is not None and self.center_weight > 0.0:
            idx_median = torch.argmin(torch.abs(self.quantiles - 0.5))
            pred_median = preds[:, idx_median]  # (B,)
            if center_target.dim() == 2:
                center_target = center_target.squeeze(1)

            diff = torch.abs(pred_median - center_target)
            soft_penalty = torch.relu(diff - self.center_eps)  # Apply slack
            center_penalty = soft_penalty.pow(2)  # Quadratic penalty if outside slack zone

            if self.reduction == 'mean':
                center_penalty = center_penalty.mean()
            elif self.reduction == 'sum':
                center_penalty = center_penalty.sum()

            loss = loss + self.center_weight * center_penalty

        return loss