# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios.
Function form: L(N) = A * (N + N0)^(-α) + B
Parameters: A, α, B, N0 (4 total)

Features:
- Robust parameter initialization via log-log fit
- Global search with differential evolution + local refinement (L-BFGS-B)
- Bounds to ensure numerical stability
- Handles edge cases (very small/large N)
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

def scaling_law_func(data_points, params):
    """
    Predict loss from training data size via a 4-parameter scaling law.
    
    L(N) = A * (N + N0)^(-alpha) + B

    Args:
        data_points: array-like of training data sizes N
        params: [A, alpha, B, N0]
    Returns:
        Predicted loss values, shape same as data_points
    """
    A, alpha, B, N0 = params
    x = np.asarray(data_points, dtype=float)
    # ensure positive argument for power
    x_shift = x + max(N0, 1e-8)
    return A * np.power(x_shift, -alpha) + B

def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law parameters to observed losses.
    
    Args:
        data_points: array-like of training sizes N
        loss_values: array-like of observed losses
    
    Returns:
        params_opt: optimized [A, alpha, B, N0]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    # basic bounds: A>0, alpha>0, B>=0, N0>=0
    max_x, max_y = max(np.max(x), 1e-8), max(np.max(y), 1e-8)
    bounds = [
        (1e-8, 10 * max_y),    # A
        (1e-8, 10.0),          # alpha
        (0.0, max_y),          # B
        (1e-8, max_x)          # N0
    ]
    # initial guess via log-log linear fit for alpha, B as min(y), A from first point
    mask = (x > 0) & (y > 0)
    if mask.sum() >= 2:
        logx, logy = np.log(x[mask]), np.log(y[mask])
        slope, intercept = np.polyfit(logx, logy, 1)
        alpha0 = max(-slope, 1e-2)
    else:
        alpha0 = 0.5
    B0 = max(min(y) * 0.9, 0.0)
    A0 = max((np.max(y) - B0) * ( (x.min() + 1e-8)**alpha0 ), 1e-3)
    N0_0 = 1e-8
    initial = [A0, alpha0, B0, N0_0]

    def objective(params):
        pred = scaling_law_func(x, params)
        return np.mean((pred - y) ** 2)

    # Global optimization
    try:
        de_result = differential_evolution(
            objective, bounds,
            strategy='best1bin',
            maxiter=300, popsize=15,
            tol=1e-6, disp=False
        )
        p0 = de_result.x
    except Exception:
        p0 = initial

    # Local refinement
    try:
        local = minimize(
            objective, p0,
            method='L-BFGS-B', bounds=bounds,
            options={'ftol':1e-9, 'maxiter':500}
        )
        params_opt = local.x if local.success else p0
    except Exception:
        params_opt = p0

    return params_opt

# Expose number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END