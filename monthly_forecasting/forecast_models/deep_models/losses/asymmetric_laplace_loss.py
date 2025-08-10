import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AsymmetricLaplaceLoss(nn.Module):
    """
    Asymmetric Laplace loss for uncertainty quantification.

    The Asymmetric Laplace distribution is parameterized by:
    - mu: location parameter (mode/median)
    - b: scale parameter (dispersion)
    - tau: asymmetry parameter (0 < tau < 1)

    This loss is particularly useful for:
    - Quantile regression (when tau != 0.5)
    - Uncertainty quantification in neural networks
    - Modeling skewed prediction errors
    """

    def __init__(self, tau: float = 0.5, reduction: str = "mean"):
        """
        Initialize Asymmetric Laplace loss.

        Args:
            tau: Asymmetry parameter (0 < tau < 1)
                - tau = 0.5: symmetric (equivalent to standard Laplace)
                - tau < 0.5: left-skewed
                - tau > 0.5: right-skewed
            reduction: Loss reduction ('mean', 'sum', 'none')
        """
        super().__init__()

        if not 0 < tau < 1:
            raise ValueError(f"tau must be between 0 and 1, got {tau}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.tau = tau
        self.reduction = reduction

        # Precompute constants for efficiency
        self.tau_factor = tau * (1 - tau)
        self.log_normalization = torch.log(torch.tensor(self.tau_factor))

        logger.info(
            f"AsymmetricLaplaceLoss initialized with tau={tau}, reduction={reduction}"
        )

    def forward(
        self, mu: torch.Tensor, b: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Asymmetric Laplace loss.

        Args:
            mu: Location parameter (predicted values) - shape: (batch_size,)
            b: Scale parameter (predicted uncertainty) - shape: (batch_size,)
            targets: Ground truth targets - shape: (batch_size,)

        Returns:
            Asymmetric Laplace loss
        """
        # Ensure all tensors have the same shape
        if mu.shape != targets.shape:
            raise ValueError(
                f"mu and targets must have same shape, got {mu.shape} and {targets.shape}"
            )
        if b.shape != targets.shape:
            raise ValueError(
                f"b and targets must have same shape, got {b.shape} and {targets.shape}"
            )

        # Ensure b is positive (add small epsilon for numerical stability)
        b = torch.clamp(b, min=1e-6)

        # Compute the residual
        residual = targets - mu

        # Asymmetric Laplace loss components
        # L(y, mu, b, tau) = (1/b) * rho_tau(y - mu) + log(b / (tau * (1 - tau)))
        # where rho_tau(u) = u * (tau - I(u < 0))

        # Compute rho_tau(residual)
        indicator = (residual >= 0).float()  # I(residual >= 0)
        rho_tau = residual * (self.tau - (1 - indicator))

        # Compute the loss
        loss = rho_tau / b + torch.log(b) - self.log_normalization

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def negative_log_likelihood(
        self, mu: torch.Tensor, b: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood of Asymmetric Laplace distribution.

        This is equivalent to the forward method but with clearer naming
        for probabilistic interpretation.

        Args:
            mu: Location parameter
            b: Scale parameter
            targets: Ground truth targets

        Returns:
            Negative log-likelihood
        """
        return self.forward(mu, b, targets)

    def predict_quantile(
        self, mu: torch.Tensor, b: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        """
        Predict specific quantile from Asymmetric Laplace parameters.

        Args:
            mu: Location parameter
            b: Scale parameter
            quantile: Desired quantile (0 < quantile < 1)

        Returns:
            Predicted quantile values
        """
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be between 0 and 1, got {quantile}")

        # Ensure b is positive
        b = torch.clamp(b, min=1e-6)

        # Quantile function for Asymmetric Laplace distribution
        scale = b / self.tau_factor

        if quantile < self.tau:
            # Lower tail
            q_pred = mu + scale * self.tau * torch.log(
                torch.tensor(2 * quantile / self.tau)
            )
        else:
            # Upper tail
            q_pred = mu - scale * (1 - self.tau) * torch.log(
                torch.tensor(2 * (1 - quantile) / (1 - self.tau))
            )

        return q_pred

    def predict_interval(
        self, mu: torch.Tensor, b: torch.Tensor, confidence: float = 0.9
    ) -> tuple:
        """
        Predict confidence interval from Asymmetric Laplace parameters.

        Args:
            mu: Location parameter
            b: Scale parameter
            confidence: Confidence level (0 < confidence < 1)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

        alpha = 1 - confidence
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower_bound = self.predict_quantile(mu, b, lower_quantile)
        upper_bound = self.predict_quantile(mu, b, upper_quantile)

        return lower_bound, upper_bound

    def compute_crps(
        self, mu: torch.Tensor, b: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Continuous Ranked Probability Score (CRPS) for Asymmetric Laplace.

        CRPS is a proper scoring rule for probabilistic forecasts.

        Args:
            mu: Location parameter
            b: Scale parameter
            targets: Ground truth targets

        Returns:
            CRPS values
        """
        # Ensure b is positive
        b = torch.clamp(b, min=1e-6)

        # Standardized residual
        z = (targets - mu) / b

        # CRPS for Asymmetric Laplace (analytical formula)
        # This is a simplified approximation - full formula is more complex
        scale = b / self.tau_factor

        # Approximate CRPS using the location-scale property
        crps = scale * (
            torch.abs(z) + (2 * self.tau - 1) * z - self.tau * (1 - self.tau)
        )

        if self.reduction == "mean":
            return crps.mean()
        elif self.reduction == "sum":
            return crps.sum()
        else:
            return crps


class AdaptiveAsymmetricLaplaceLoss(AsymmetricLaplaceLoss):
    """
    Adaptive Asymmetric Laplace loss with learnable tau parameter.

    Allows the asymmetry parameter to be learned during training,
    which can be useful when the optimal quantile is unknown.
    """

    def __init__(
        self,
        initial_tau: float = 0.5,
        learnable_tau: bool = True,
        tau_bounds: tuple = (0.01, 0.99),
        reduction: str = "mean",
    ):
        """
        Initialize adaptive Asymmetric Laplace loss.

        Args:
            initial_tau: Initial asymmetry parameter
            learnable_tau: Whether tau is a learnable parameter
            tau_bounds: Bounds for tau parameter (min, max)
            reduction: Loss reduction
        """
        # Don't call super().__init__ as we handle tau differently
        nn.Module.__init__(self)

        if not 0 < initial_tau < 1:
            raise ValueError(f"initial_tau must be between 0 and 1, got {initial_tau}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.learnable_tau = learnable_tau
        self.tau_bounds = tau_bounds
        self.reduction = reduction

        if learnable_tau:
            # Convert tau to logits for unconstrained optimization
            initial_logit = torch.log(torch.tensor(initial_tau) / (1 - initial_tau))
            self.tau_logit = nn.Parameter(initial_logit)
        else:
            self.register_buffer(
                "tau_logit", torch.log(torch.tensor(initial_tau) / (1 - initial_tau))
            )

        # Initialize tau-dependent constants
        current_tau = self.tau
        self.tau_factor = current_tau * (1 - current_tau)
        self.log_normalization = torch.log(torch.tensor(self.tau_factor))

        logger.info(
            f"AdaptiveAsymmetricLaplaceLoss initialized with learnable_tau={learnable_tau}"
        )

    @property
    def tau(self):
        """Get current tau value."""
        # Convert logit back to tau with bounds
        min_tau, max_tau = self.tau_bounds
        raw_tau = torch.sigmoid(self.tau_logit)
        tau = raw_tau * (max_tau - min_tau) + min_tau
        return tau.item() if tau.numel() == 1 else tau

    @tau.setter
    def tau(self, value):
        """Set tau value (for compatibility)."""
        if self.learnable_tau:
            logit = torch.log(torch.tensor(value) / (1 - value))
            self.tau_logit.data = logit
        else:
            raise ValueError("Cannot set tau when learnable_tau=False")

    def forward(
        self, mu: torch.Tensor, b: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive Asymmetric Laplace loss.

        Args:
            mu: Location parameter
            b: Scale parameter
            targets: Ground truth targets

        Returns:
            Loss value
        """
        # Update tau-dependent constants
        current_tau = self.tau
        self.tau_factor = current_tau * (1 - current_tau)
        self.log_normalization = torch.log(torch.tensor(self.tau_factor))

        # Call parent forward method
        return super().forward(mu, b, targets)


class RobustAsymmetricLaplaceLoss(AsymmetricLaplaceLoss):
    """
    Robust version of Asymmetric Laplace loss with outlier handling.

    Includes mechanisms to handle outliers and extreme values that
    might destabilize training.
    """

    def __init__(
        self,
        tau: float = 0.5,
        reduction: str = "mean",
        clip_threshold: Optional[float] = None,
        huber_delta: Optional[float] = None,
    ):
        """
        Initialize robust Asymmetric Laplace loss.

        Args:
            tau: Asymmetry parameter
            reduction: Loss reduction
            clip_threshold: Threshold for gradient clipping (optional)
            huber_delta: Delta for Huber-like robustness (optional)
        """
        super().__init__(tau=tau, reduction=reduction)

        self.clip_threshold = clip_threshold
        self.huber_delta = huber_delta

        logger.info(
            f"RobustAsymmetricLaplaceLoss initialized with "
            f"clip_threshold={clip_threshold}, huber_delta={huber_delta}"
        )

    def forward(
        self, mu: torch.Tensor, b: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute robust Asymmetric Laplace loss.

        Args:
            mu: Location parameter
            b: Scale parameter
            targets: Ground truth targets

        Returns:
            Robust loss value
        """
        # Ensure all tensors have the same shape
        if mu.shape != targets.shape or b.shape != targets.shape:
            raise ValueError("All tensors must have the same shape")

        # Ensure b is positive and bounded
        b = torch.clamp(b, min=1e-6, max=1e6)  # Prevent extreme scale values

        # Compute residual
        residual = targets - mu

        # Apply Huber-like robustness if specified
        if self.huber_delta is not None:
            # Smooth transition for large residuals
            abs_residual = torch.abs(residual)
            huber_mask = abs_residual <= self.huber_delta

            smooth_residual = torch.where(
                huber_mask,
                residual,
                torch.sign(residual)
                * (self.huber_delta + torch.log(1 + abs_residual - self.huber_delta)),
            )
            residual = smooth_residual

        # Standard Asymmetric Laplace computation
        indicator = (residual >= 0).float()
        rho_tau = residual * (self.tau - (1 - indicator))

        loss = rho_tau / b + torch.log(b) - self.log_normalization

        # Apply gradient clipping if specified
        if self.clip_threshold is not None:
            loss = torch.clamp(loss, -self.clip_threshold, self.clip_threshold)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
