# EVOLVE-BLOCK-START
"""
Enhanced scaling law discovery for LLM finetuning scenarios.

We fit a 4-parameter power-law with shift and floor:
    loss(x) = a * (x + c)^(-b) + d

Where:
    a > 0      -- scale coefficient
    b > 0      -- power-law exponent
    c >= 0     -- horizontal shift to improve numeric stability
    d         -- asymptotic loss floor

Fitting uses a global search (differential evolution) followed by
local refinement (L-BFGS-B).  Robust bounds and initialization
improve convergence and generalization.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    Predict loss according to a 4-parameter scaling law:
        L(x) = a * (x + c)^(-b) + d

    Args:
        data_points: array-like, training data sizes (must be >= 0)
        params: array-like of length 4: [a, b, c, d]

    Returns:
        numpy array of predicted losses
    """
    x = np.asarray(data_points, dtype=np.float64)
    a, b, c, d = params
    # ensure positivity inside power
    xp = x + max(c, 0.0) + 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        y = a * np.power(xp, -b) + d
    # clip any NaNs or infs to large values
    y = np.where(np.isfinite(y), y, np.finfo(np.float64).max)
    return y

# declare number of params for downstream checks
scaling_law_func.num_params = 4

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (data_points, loss_values).

    Args:
        data_points: array-like of training data sizes
        loss_values: array-like of observed losses

    Returns:
        numpy array of optimized [a, b, c, d]
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # basic statistics
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    y_range = max(y_max - y_min, 1e-6)
    # Parameter bounds: [a, b, c, d]
    bounds = [
        (1e-8, 10.0 * y_range),   # a: scale
        (1e-6, 5.0),              # b: exponent
        (0.0, x_max * 2.0),        # c: shift
        (y_min - y_range, y_max + y_range)  # d: floor (can be negative)
    ]
    # initial guess
    init_guess = np.array([
        y_range,      # a ~ range of y
        0.5,          # b ~ 0.5
        x_min * 0.1,  # c small shift
        y_min         # d ~ minimal observed loss
    ], dtype=np.float64)

    # Objective: mean squared error
    def obj_fn(params):
        pred = scaling_law_func(x, params)
        if not np.all(np.isfinite(pred)):
            return 1e6
        return np.mean((pred - y) ** 2)

    # 1) Global search: differential evolution
    try:
        result_de = differential_evolution(
            obj_fn, bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False,
            seed=42
        )
        best = result_de.x
    except Exception:
        best = init_guess

    # 2) Local refinement: L-BFGS-B
    try:
        result_local = minimize(
            obj_fn, best,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol':1e-9, 'gtol':1e-6, 'maxiter':1000}
        )
        if result_local.success:
            best = result_local.x
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END