# EVOLVE-BLOCK-START
"""
Improved scaling law discovery for LLM finetuning scenarios.
Uses a 4-parameter form:
    L(N) = a * (N + d)^(-b) + c
with bounds and multi-start L-BFGS-B optimization for robustness.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Scaling law function: L(N) = a * (N + d)^(-b) + c

    Args:
        data_points: array-like of training data sizes
        params: sequence of 4 parameters [a, b, c, d]
                a > 0, b > 0, c >= 0, d >= 0

    Returns:
        np.ndarray of predicted loss values
    """
    x = np.asarray(data_points, dtype=float)
    a, b, c, d = params
    # shift and ensure positivity
    x_safe = x + d + 1e-8
    return a * (x_safe ** (-b)) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to data.

    Args:
        data_points: array-like of training data sizes
        loss_values: array-like of observed loss values

    Returns:
        np.ndarray of optimized parameters [a, b, c, d]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Ensure strictly positive losses for numerical stability
    y_min = np.min(y)
    if y_min <= 0:
        y = y - y_min + 1e-6

    # Objective: mean squared error
    def objective(params):
        a, b, c, d = params
        # enforce simple bounds in the objective
        if a <= 0 or b <= 0 or c < 0 or d < 0:
            return np.inf
        y_pred = scaling_law_func(x, params)
        return np.mean((y_pred - y) ** 2)

    # Build a few smart initial guesses
    y_range = np.max(y) - np.min(y)
    x_med = np.median(x)
    inits = [
        [y_range + 1e-3, 0.5, np.min(y), x_med],
        [np.max(y),         0.3, 0.5 * np.min(y), x_med * 0.1],
        [y_range,           1.0, 0.0,             1e2]
    ]

    # Parameter bounds: a>0, b>0, c>=0, d>=0
    bounds = [(1e-8, None), (1e-8, None), (0, None), (0, None)]

    best_loss = np.inf
    best_params = None

    # Multi-start optimization
    for init in inits:
        try:
            res = minimize(
                objective,
                x0=init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-9}
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            continue

    # Fallback to a simple log-linear fit if all starts fail
    if best_params is None:
        # Fit log(y - y_min + eps) = log(a) - b * log(x + 1)
        eps = 1e-8
        xp = np.log(x + 1.0)
        yp = np.log(y - np.min(y) + eps)
        A = np.vstack([xp, np.ones_like(xp)]).T
        slope, intercept = np.linalg.lstsq(A, yp, rcond=None)[0]
        b_est = -slope
        a_est = np.exp(intercept)
        c_est = np.min(y)
        d_est = 1.0
        best_params = np.array([a_est, b_est, c_est, d_est], dtype=float)

    return np.array(best_params, dtype=float)

# Declare the expected number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END