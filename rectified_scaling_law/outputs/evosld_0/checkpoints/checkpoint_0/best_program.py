# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios
Uses a 4-parameter form: L(N) = a * (N + d)^(-b) + c
with robust multi-start fitting under bounds to ensure numerical stability
and good generalization.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Scaling law: L(N) = a * (N + d)^(-b) + c
    Params:
      a > 0  (amplitude)
      b > 0  (exponent)
      c >= 0 (floor)
      d >= 0 (offset)
    """
    # unpack and enforce positivity for stability
    a, b, c, d = params
    a = np.maximum(a, 1e-12)
    b = np.maximum(b, 1e-12)
    c = np.maximum(c, 0.0)
    d = np.maximum(d, 0.0)

    x = np.asarray(data_points, dtype=float)
    # compute prediction
    return a * np.power(x + d, -b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (data_points, loss_values)
    using bounded L-BFGS-B and multi-start initialization.
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Precompute stats for sensible inits
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    x_mean = np.mean(x)

    # Multi-start initial guesses
    inits = [
        # large amplitude, moderate decay, floor at min(y), offset at mean(N)
        [y_range, 0.5, y_min, x_mean],
        # amplitude ~ max_loss, stronger decay, zero floor, small offset
        [y_max, 1.0, 0.0, 1e3],
        # unity amplitude, shallow decay, floor at min(y), offset moderate
        [1.0, 0.1, y_min, 1e4],
        # amplitude at first point, mid decay, no offset
        [y[0], 0.7, 0.0, 0.0],
        # small decay, no floor, small offset
        [y_range, 0.2, 0.0, 100.0]
    ]

    # Parameter bounds: a>0, b>0, c>=0, d>=0
    bounds = [(1e-12, None), (1e-12, None), (0.0, None), (0.0, None)]

    def objective(p):
        pred = scaling_law_func(x, p)
        # MSE loss
        return np.mean((pred - y) ** 2)

    best_params = None
    best_loss = np.inf

    # Run multi-start L-BFGS-B
    for init in inits:
        try:
            res = minimize(
                objective,
                x0=np.asarray(init, dtype=float),
                bounds=bounds,
                method='L-BFGS-B',
                options={'ftol': 1e-12, 'maxiter': 5000}
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            continue

    # Fallback to 2-param power law fit if all starts fail
    if best_params is None:
        # log-linear regression: log(y) ≈ log(a) - b log(x)
        mask = (x > 0) & (y > 0)
        if np.sum(mask) >= 2:
            lx = np.log(x[mask])
            ly = np.log(y[mask])
            slope, intercept = np.polyfit(lx, ly, 1)
            b_est = -slope
            a_est = np.exp(intercept)
        else:
            a_est, b_est = 1.0, 0.5
        best_params = np.array([a_est, b_est, 0.0, 0.0], dtype=float)

    return np.asarray(best_params, dtype=float)

# Specify number of parameters used
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END