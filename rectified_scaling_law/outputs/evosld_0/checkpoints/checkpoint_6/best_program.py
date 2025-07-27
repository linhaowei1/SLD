# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios.
We use a 4‐parameter form:
    L(N) = a * (N + N0)^(-b) + c
with robust initialization and bounded L-BFGS-B optimization.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A 4-parameter scaling law:
        L = a * (N + N0)^(-b) + c

    Args:
        data_points: array-like of training data sizes (N)
        params: array-like of 4 parameters [a, b, c, N0]
    Returns:
        loss predictions array
    """
    a, b, c, N0 = params
    x = np.asarray(data_points, dtype=np.float64)
    # ensure positivity and numerical stability
    x_eff = x + np.abs(N0) + 1e-12
    return a * np.power(x_eff, -b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to observed (N, loss) pairs.

    Args:
        data_points: array-like of training data sizes (N)
        loss_values: array-like of observed losses
    Returns:
        best-fit parameters [a, b, c, N0]
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)

    # Floor c0 slightly below the smallest observed loss to allow a positive tail
    c0 = max(y.min() * 0.9, 1e-8)

    # Two candidate offsets for N0 to seed the fit
    N0_guesses = [1.0, np.median(x) * 0.1]

    best_mse = np.inf
    best_params = None

    # Objective: mean squared error
    def mse_obj(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2)

    # Bounds: a>0, b>0, c>=0, 0<=N0<=max(x)
    bounds = [(1e-8, None), (1e-8, 10.0), (0.0, None), (0.0, x.max())]

    for N0_init in N0_guesses:
        # Linearize log(loss - c0) ≈ log(a) - b * log(x + N0)
        x_eff = x + N0_init
        y_adj = np.clip(y - c0, 1e-8, None)
        logx = np.log(x_eff)
        logy = np.log(y_adj)

        # Simple linear regression in log-log space
        slope, intercept = np.polyfit(logx, logy, 1)
        b_init = -slope
        a_init = np.exp(intercept)

        # Try a few scaled initializations
        for scale in (0.5, 1.0, 2.0):
            init = np.array([a_init * scale,
                             b_init * scale,
                             c0,
                             N0_init * scale],
                            dtype=np.float64)
            # Clip to bounds
            for i, (low, high) in enumerate(bounds):
                if init[i] < low:
                    init[i] = low * 1.1
                elif high is not None and init[i] > high:
                    init[i] = high * 0.9

            res = minimize(mse_obj,
                           init,
                           method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success and res.fun < best_mse:
                best_mse = res.fun
                best_params = res.x

    # Fallback to a simple heuristic if optimization failed
    if best_params is None:
        best_params = np.array([1.0, 0.5, c0, 1.0], dtype=np.float64)

    return best_params

# Declare number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END