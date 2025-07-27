# EVOLVE-BLOCK-START
"""
Refined scaling law discovery for LLM finetuning scenarios.

We keep the 4-parameter form:
    L(N) = a * (N + N0)^(-b) + c

Parameters:
  a  > 0   scale coefficient
  b  > 0   power-law exponent
  N0 ≥ 0   horizontal shift
  c  ≥ 0   asymptotic loss floor

Fitting pipeline:
 1) Preprocess data, clip to avoid zeros.
 2) Global search (differential evolution) on log‐MSE.
 3) Local refinement on log‐MSE (L-BFGS-B).
 4) Final polish on linear MSE (L-BFGS-B).
 5) Return best parameters.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law:
        L(N) = a * (N + N0)^(-b) + c

    Args:
      data_points: array-like of training data sizes
      params: length-4 array [a, b, N0, c]

    Returns:
      pred: array of predicted loss values
    """
    x = np.asarray(data_points, dtype=np.float64)
    a, b, N0, c = params
    # enforce positivity
    a = np.maximum(a, 1e-12)
    b = np.maximum(b, 1e-12)
    N0 = np.maximum(N0, 0.0)
    c = np.maximum(c, 0.0)
    return a * (x + N0) ** (-b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (N, loss) data.

    Args:
      data_points: array-like of training data sizes
      loss_values: array-like of observed losses

    Returns:
      best_params: length-4 array [a, b, N0, c]
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # clip to avoid zeros and negatives
    eps = 1e-12
    y = np.clip(y, eps, None)

    # parameter bounds
    bounds = [
        (1e-8, np.max(y) * 10.0),      # a
        (1e-8, 10.0),                  # b
        (0.0,   np.max(x)),            # N0
        (0.0,   np.min(y))             # c
    ]

    # Objective 1: log-space MSE
    def obj_log(params):
        pred = scaling_law_func(x, params)
        if np.any(pred <= 0):
            return np.inf
        return np.mean((np.log(pred + eps) - np.log(y)) ** 2)

    # Objective 2: linear MSE
    def obj_lin(params):
        pred = scaling_law_func(x, params)
        return np.mean((pred - y) ** 2)

    # 1) Global search on log‐MSE
    de_result = differential_evolution(
        obj_log, bounds,
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=1e-6,
        polish=False,
        seed=1234
    )
    x0 = de_result.x

    # 2) Local refinement on log‐MSE
    loc1 = minimize(
        obj_log, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-9, 'gtol':1e-8, 'maxiter':1000}
    )
    p1 = loc1.x if loc1.success else x0

    # 3) Final polish on linear MSE
    loc2 = minimize(
        obj_lin, p1,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-12, 'gtol':1e-9, 'maxiter':1000}
    )
    best_params = loc2.x if loc2.success else p1

    return best_params

# metadata
scaling_law_func.num_params = 4

# EVOLVE-BLOCK-END