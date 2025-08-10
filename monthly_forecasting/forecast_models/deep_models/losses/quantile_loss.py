import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.

    Computes the quantile loss for multiple quantiles simultaneously,
    enabling uncertainty quantification in forecasting models.
    """

    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize quantile loss.

        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()

        # Validate quantiles
        for q in quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantiles must be between 0 and 1, got {q}")

        self.quantiles = sorted(quantiles)
        self.num_quantiles = len(quantiles)

        # Register quantiles as buffer so they move with the model
        self.register_buffer("quantile_levels", torch.tensor(quantiles))

        logger.info(f"QuantileLoss initialized with quantiles: {quantiles}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantiles (batch_size, num_quantiles)
            targets: Ground truth targets (batch_size, 1) or (batch_size,)

        Returns:
            Quantile loss (scalar)
        """
        if predictions.dim() != 2 or predictions.size(1) != self.num_quantiles:
            raise ValueError(
                f"Predictions must have shape (batch_size, {self.num_quantiles}), "
                f"got {predictions.shape}"
            )

        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)  # (batch_size, 1)
        elif targets.dim() == 2 and targets.size(1) != 1:
            raise ValueError(
                f"Targets must have shape (batch_size, 1), got {targets.shape}"
            )

        batch_size = predictions.size(0)

        # Expand targets to match quantile predictions: (batch_size, num_quantiles)
        targets_expanded = targets.expand(-1, self.num_quantiles)

        # Compute quantile loss for each quantile
        errors = targets_expanded - predictions  # (batch_size, num_quantiles)

        # Quantile loss: max(tau * error, (tau - 1) * error)
        # This is equivalent to: tau * error if error >= 0, else (tau - 1) * error
        quantile_levels = self.quantile_levels.expand(
            batch_size, -1
        )  # (batch_size, num_quantiles)

        loss = torch.where(
            errors >= 0, quantile_levels * errors, (quantile_levels - 1) * errors
        )

        # Average over batch and quantiles
        return loss.mean()

    def forward_single_quantile(
        self, predictions: torch.Tensor, targets: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        """
        Compute quantile loss for a single quantile.

        Args:
            predictions: Predicted values (batch_size,)
            targets: Ground truth targets (batch_size,)
            quantile: Quantile level (0 < quantile < 1)

        Returns:
            Quantile loss (scalar)
        """
        if not 0 < quantile < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")

        errors = targets - predictions
        loss = torch.where(errors >= 0, quantile * errors, (quantile - 1) * errors)

        return loss.mean()

    def compute_interval_score(
        self, predictions: torch.Tensor, targets: torch.Tensor, alpha: float = 0.1
    ) -> torch.Tensor:
        """
        Compute interval score for prediction intervals.

        Args:
            predictions: Predicted quantiles (batch_size, num_quantiles)
            targets: Ground truth targets (batch_size,)
            alpha: Significance level (e.g., 0.1 for 90% prediction intervals)

        Returns:
            Interval score (scalar)
        """
        if self.num_quantiles < 2:
            raise ValueError("Need at least 2 quantiles to compute interval score")

        # Find lower and upper quantiles
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        # Find indices of quantiles closest to desired levels
        lower_idx = min(
            range(self.num_quantiles),
            key=lambda i: abs(self.quantiles[i] - lower_quantile),
        )
        upper_idx = min(
            range(self.num_quantiles),
            key=lambda i: abs(self.quantiles[i] - upper_quantile),
        )

        lower_pred = predictions[:, lower_idx]
        upper_pred = predictions[:, upper_idx]

        # Interval score components
        interval_width = upper_pred - lower_pred

        # Penalty for being below lower bound
        lower_violation = torch.relu(lower_pred - targets) * (2 / alpha)

        # Penalty for being above upper bound
        upper_violation = torch.relu(targets - upper_pred) * (2 / alpha)

        # Total interval score
        interval_score = interval_width + lower_violation + upper_violation

        return interval_score.mean()


class AdaptiveQuantileLoss(QuantileLoss):
    """
    Adaptive quantile loss that adjusts quantile levels during training.

    Useful for automatically learning appropriate quantile levels for
    uncertainty quantification.
    """

    def __init__(
        self,
        initial_quantiles: List[float] = [0.1, 0.5, 0.9],
        learnable: bool = True,
        quantile_bounds: tuple = (0.01, 0.99),
    ):
        """
        Initialize adaptive quantile loss.

        Args:
            initial_quantiles: Initial quantile levels
            learnable: Whether quantile levels are learnable parameters
            quantile_bounds: Bounds for quantile levels (min, max)
        """
        # Don't call super().__init__ as we'll handle quantiles differently
        nn.Module.__init__(self)

        self.learnable = learnable
        self.quantile_bounds = quantile_bounds

        if learnable:
            # Make quantiles learnable parameters
            self.raw_quantiles = nn.Parameter(
                torch.tensor(initial_quantiles, dtype=torch.float32)
            )
        else:
            # Fixed quantiles
            self.register_buffer(
                "raw_quantiles", torch.tensor(initial_quantiles, dtype=torch.float32)
            )

        self.num_quantiles = len(initial_quantiles)

        logger.info(f"AdaptiveQuantileLoss initialized with learnable={learnable}")

    @property
    def quantiles(self):
        """Get current quantile levels."""
        if self.learnable:
            # Apply sigmoid to keep quantiles in valid range
            min_q, max_q = self.quantile_bounds
            quantiles = torch.sigmoid(self.raw_quantiles) * (max_q - min_q) + min_q
            return quantiles.sort()[0]  # Ensure sorted order
        else:
            return self.raw_quantiles

    @property
    def quantile_levels(self):
        """Get quantile levels as tensor."""
        return self.quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive quantile loss.

        Args:
            predictions: Predicted quantiles (batch_size, num_quantiles)
            targets: Ground truth targets (batch_size, 1) or (batch_size,)

        Returns:
            Quantile loss (scalar)
        """
        if predictions.dim() != 2 or predictions.size(1) != self.num_quantiles:
            raise ValueError(
                f"Predictions must have shape (batch_size, {self.num_quantiles}), "
                f"got {predictions.shape}"
            )

        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        batch_size = predictions.size(0)

        # Get current quantile levels
        current_quantiles = self.quantile_levels

        # Expand targets and quantiles
        targets_expanded = targets.expand(-1, self.num_quantiles)
        quantile_levels = current_quantiles.expand(batch_size, -1)

        # Compute quantile loss
        errors = targets_expanded - predictions
        loss = torch.where(
            errors >= 0, quantile_levels * errors, (quantile_levels - 1) * errors
        )

        return loss.mean()


class PinballLoss(nn.Module):
    """
    Pinball loss (another name for quantile loss).

    Provided for compatibility and clarity in naming.
    """

    def __init__(self, quantile: float = 0.5):
        """
        Initialize pinball loss for a single quantile.

        Args:
            quantile: Quantile level (0 < quantile < 1)
        """
        super().__init__()

        if not 0 < quantile < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")

        self.quantile = quantile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss.

        Args:
            predictions: Predicted values (batch_size,)
            targets: Ground truth targets (batch_size,)

        Returns:
            Pinball loss (scalar)
        """
        errors = targets - predictions
        loss = torch.where(
            errors >= 0, self.quantile * errors, (self.quantile - 1) * errors
        )

        return loss.mean()
