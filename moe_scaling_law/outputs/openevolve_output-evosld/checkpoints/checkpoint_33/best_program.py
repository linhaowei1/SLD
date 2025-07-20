# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models

We model validation loss via a 6-parameter log–quadratic form in the
(log num_experts, log total_params) feature space. This yields
numerical stability and a closed-form initialization via linear
regression, followed by nonlinear refinement.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict validation loss given MoE configuration.

    We use
        x = log(E + eps)
        y = log(P + eps)
        log_loss = θ0
                 + θ1 * x
                 + θ2 * y
                 + θ3 * x^2
                 + θ4 * y^2
                 + θ5 * x * y
        loss = exp(log_loss)

    Args:
        num_experts: array-like, number of experts (E)
        total_parameter_count: array-like, total parameters (P)
        params: 6-element array [θ0, θ1, θ2, θ3, θ4, θ5]

    Returns:
        pred_loss: array-like, predicted losses
    """
    θ0, θ1, θ2, θ3, θ4, θ5 = params
    E = np.maximum(num_experts, 0.0) + 1e-8
    P = np.maximum(total_parameter_count, 0.0) + 1e-8
    x = np.log(E)
    y = np.log(P)
    log_loss = θ0 + θ1 * x + θ2 * y + θ3 * x**2 + θ4 * y**2 + θ5 * x * y
    return np.exp(log_loss)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log–quadratic MoE scaling law.

    We first solve a linear least-squares for log-loss in the
    basis [1, x, y, x^2, y^2, x*y], then refine via
    nonlinear least-squares on the original-loss residuals.
    """
    # Convert to arrays
    E = np.asarray(num_experts, dtype=np.float64)
    P = np.asarray(total_parameter_count, dtype=np.float64)
    L = np.asarray(loss_values, dtype=np.float64)

    # Build features in log-space
    eps = 1e-8
    x = np.log(np.maximum(E, 0.0) + eps)
    y = np.log(np.maximum(P, 0.0) + eps)

    # Target for linear init
    t = np.log(np.maximum(L, eps))

    # Design matrix: [1, x, y, x^2, y^2, x*y]
    X = np.vstack((np.ones_like(x), x, y, x**2, y**2, x*y)).T

    # Closed-form initial guess via linear regression
    θ_init, *_ = np.linalg.lstsq(X, t, rcond=None)

    # Refine parameters by minimizing residuals in original loss space
    def residuals(θ):
        pred = np.exp(X.dot(θ))
        return pred - L

    result = least_squares(
        residuals,
        θ_init,
        method='lm',
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12
    )

    θ_opt = result.x if result.success else θ_init
    return θ_opt

# Expose the number of parameters
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END