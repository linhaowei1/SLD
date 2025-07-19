# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios.
Improved version with a 4-parameter power-law-plus-offset form:
    Loss(N) = A * (N + c)^{-alpha} + B
Fits parameters [A, alpha, c, B] via multi-start L-BFGS-B to minimize NMSE.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict loss given data sizes and 4 scaling-law parameters.
    Loss(N) = A * (N + c)^(-alpha) + B

    Args:
        data_points: 1D array of training data sizes (N).
        params: Array-like of 4 parameters [A, alpha, c, B].

    Returns:
        1D numpy array of predicted losses.
    """
    A, alpha, c, B = params
    x = np.asarray(data_points, dtype=float)
    # ensure numeric stability
    x_c = x + np.abs(c)
    return A * np.power(x_c, -np.abs(alpha)) + B

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to data_points and loss_values by minimizing NMSE.
    Uses multiple random restarts and L-BFGS-B with simple bounds for stability.

    Args:
        data_points: 1D array of training data sizes.
        loss_values: 1D array of observed losses.

    Returns:
        best_params: array of optimized [A, alpha, c, B].
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    var_y = np.var(y) if np.var(y) > 0 else 1.0

    # Objective: normalized MSE
    def obj(params):
        pred = scaling_law_func(x, params)
        return np.mean((pred - y) ** 2) / var_y

    # Bounds: A>=1e-12, alpha>=1e-12, c>=0, B>=0
    bnds = [(1e-12, None), (1e-12, None), (0.0, None), (0.0, None)]

    # Prepare multiple starts
    starts = []
    # Deterministic start based on data
    x_min, x_med = np.min(x), np.median(x)
    y_min, y_max = np.min(y), np.max(y)
    alpha0 = 0.5
    c0 = x_med / 10.0
    B0 = y_min
    A0 = (y_max - y_min) * ((x_min + c0) ** alpha0 + 1e-12)
    starts.append(np.array([A0, alpha0, c0, B0]))
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        a = (y_max - y_min) * rng.uniform(0.5, 2.0)
        alpha = rng.uniform(0.1, 2.0)
        c = rng.uniform(0.0, x_med)
        B = y_min + rng.uniform(0.0, 0.5 * (y_max - y_min))
        starts.append(np.array([a, alpha, c, B]))

    best_loss = np.inf
    best_params = starts[0]

    # Run optimization from each start
    for p0 in starts:
        try:
            res = minimize(obj, p0, method='L-BFGS-B', bounds=bnds,
                           options={'ftol':1e-9, 'maxiter':1000})
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            continue

    return best_params

# Attach metadata
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END