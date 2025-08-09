# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Improved four‐parameter scaling law:
    L(N) = B + A * (N + N0)^(-α)
Parameters are kept positive by optimizing in the log‐domain:
    params = [logA, logα, logN0, logB]
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict loss L given data_sizes N and log‐domain parameters.

    Args:
      data_points: array‐like of shape (N,) or (N,1) of data sizes.
      params: array‐like of 4 elements [logA, logα, logN0, logB].

    Returns:
      preds: 1D numpy array of predicted losses.
    """
    X = np.asarray(data_points).reshape(-1)
    p = np.asarray(params).reshape(-1)
    if p.size != 4:
        raise ValueError("Expected 4 parameters: [logA, logα, logN0, logB]")
    logA, log_alpha, log_N0, logB = p
    A     = np.exp(logA)
    α     = np.exp(log_alpha)
    N0    = np.exp(log_N0)
    B     = np.exp(logB)
    # Compute scaling law: plateau B plus decaying term A*(N+N0)^(-α)
    return B + A * (X + N0) ** (-α)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4‐parameter scaling law to (data_points, loss_values).

    Args:
      data_points: array‐like of shape (N,) or (N,1) of data sizes.
      loss_values: array‐like of shape (N,) of corresponding losses.

    Returns:
      p_opt: array of 4 optimized log‐domain parameters.
    """
    X = np.asarray(data_points).reshape(-1)
    y = np.asarray(loss_values).reshape(-1)

    # Natural‐domain initial guesses
    B0 = max(np.min(y), 1e-6)                           # asymptotic floor ≈ smallest loss
    # amplitude ≈ initial loss drop
    A0 = max(np.mean(y[:max(1, len(y)//3)]) - B0, 1e-6)
    α0 = 0.5                                           # typical decay exponent
    N_median = np.median(X)
    N0_0 = max(N_median * 0.1, 1.0)                    # horizontal shift

    # pack into log‐domain
    p0 = np.log([A0, α0, N0_0, B0])

    # Mean squared error objective
    def _mse(p):
        pred = scaling_law_func(X, p)
        return np.mean((pred - y) ** 2)

    # Use L-BFGS-B for robust, bound‐aware quasi-Newton
    res = minimize(
        _mse,
        p0,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "maxiter": 5000}
    )

    # Return optimized params or fallback to initial if fit failed
    return res.x if res.success else p0
# EVOLVE-BLOCK-END