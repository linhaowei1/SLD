# EVOLVE-BLOCK-START
"""
Improved scaling‐law model:
  L(N) = B + A * (N + C)^(−α)
with 4 parameters (A, α, C, B) all kept positive via an exp–reparameterization.
Fitting is done in the unconstrained log‐domain using L-BFGS-B for stability.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    data_points: array of shape (N,) or (N,1) of data sizes
    params: array of 4 real numbers [pA, pα, pC, pB] in log‐domain
    Returns: Predicted loss of shape (N,)
    """
    # Flatten input to 1D
    X = np.asarray(data_points).reshape(-1)
    p = np.asarray(params).reshape(-1)
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    # Reconstruct positive parameters
    A     = np.exp(p[0])    # amplitude
    α     = np.exp(p[1])    # exponent
    C     = np.exp(p[2])    # horizontal offset
    B     = np.exp(p[3])    # asymptotic floor
    # Compute scaling‐law prediction
    return B + A * (X + C) ** (-α)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the 4‐parameter scaling law to (data_points, loss_values).
    Returns the optimized params in the same log‐domain form accepted
    by scaling_law_func.
    """
    # Prepare 1D arrays
    X = np.asarray(data_points).reshape(-1)
    y = np.asarray(loss_values).reshape(-1)
    # Initial guesses for natural‐domain params
    #   A0 ≈ range of y, α0 ~ 0.5, C0 ~ median N, B0 ≈ min y
    A0 = max(np.max(y) - np.min(y), 1e-3)
    α0 = 0.5
    C0 = max(np.median(X), 1.0)
    B0 = max(np.min(y), 1e-3)
    # Log‐domain starting point
    p0 = np.log([A0, α0, C0, B0])

    # Objective: mean squared error
    def _obj(p):
        pred = scaling_law_func(X, p)
        return np.mean((pred - y) ** 2)

    # Use L-BFGS-B for robust, bounded quasi‐Newton
    result = minimize(_obj, p0, method="L-BFGS-B")
    p_opt = result.x if result.success else p0
    return p_opt
# EVOLVE-BLOCK-END