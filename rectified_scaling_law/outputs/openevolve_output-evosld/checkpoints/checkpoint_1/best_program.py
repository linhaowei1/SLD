# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Improved implementation using a two-term power law and robust multi-start fitting.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Two-term power-law scaling function:
        loss(x) = A * x^{-alpha} + B * x^{-beta}
    where x is the training data size.

    Args:
        data_points: array-like, training data sizes (must be > 0)
        params: array-like of length 4 [A, alpha, B, beta]
            A, B       : positive amplitudes
            alpha,beta : positive exponents

    Returns:
        numpy array of predicted losses
    """
    A, alpha, B, beta = params
    x = np.asarray(data_points, dtype=float)
    # numerical safety: enforce x >= 1
    x = np.maximum(x, 1.0)
    return A * np.power(x, -alpha) + B * np.power(x, -beta)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the two-term power-law scaling law to observed (x, loss) data.

    Uses a log-log linear fit for initialization, then a small multi-start L-BFGS-B.

    Args:
        data_points : array-like, training data sizes
        loss_values : array-like, observed loss values

    Returns:
        params: numpy array of length 4 [A, alpha, B, beta]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    # Filter out non-positive for log-fit
    mask = (x > 0) & (y > 0)
    if mask.sum() >= 2:
        # log-log linear regression: log(y) ≈ m * log(x) + c
        m, c = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
        # interpret: y ≈ exp(c) * x^m  => exponent = -m, amplitude = exp(c)
        alpha0 = -m
        A0 = np.exp(c)
    else:
        # fallback defaults
        alpha0 = 0.5
        A0 = np.max(y)
    # initialize second term as smaller amplitude and steeper exponent
    B0 = A0 * 0.1
    beta0 = max(alpha0 * 2.0, alpha0 + 0.5, 1.0)
    initial = np.array([A0, alpha0, B0, beta0], dtype=float)

    # bounds: all parameters strictly positive, exponents within [1e-6, 10]
    bounds = [
        (1e-12, None),   # A
        (1e-6,  10.0),   # alpha
        (1e-12, None),   # B
        (1e-6,  10.0)    # beta
    ]

    def mse_obj(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2)

    best_params = None
    best_loss = np.inf
    rng = np.random.RandomState(42)

    # Multi-start local optimization (first start = linear init)
    for i in range(5):
        if i == 0:
            start = initial
        else:
            # jitter initial guess by ±20%
            jitter = 1.0 + 0.2 * (rng.rand(4) - 0.5)
            start = initial * jitter

        res = minimize(
            mse_obj,
            start,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 5000,
                'ftol': 1e-9,
                'gtol': 1e-6
            }
        )

        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # If all fits fail, fall back to initial guess
    if best_params is None:
        best_params = initial

    return best_params

# Declare number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END