# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios

We adopt a 4-parameter form:
    L(N) = a * (N + N0)^(-b) + c

where
    a  > 0  scales the power-law term,
    b  > 0  is the power-law exponent,
    N0 > 0  is a shift on the data scale,
    c  ≥ 0  is an asymptotic floor.

Fitting procedure:
  1) Transform both N and losses to numpy arrays.
  2) Define a log–MSE objective to stabilize dynamic ranges.
  3) Use a global search (differential evolution) under bounds.
  4) Refine with a local quasi-Newton (L-BFGS-B).
  5) Return the best 4 parameters.

This yields robust, numerically stable fits across wide data/ loss scales.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law:
        L(N) = a * (N + N0)^(-b) + c

    Args:
      data_points: array-like, training data sizes
      params: length-4 array [a, b, N0, c]

    Returns:
      pred: array of predicted loss values
    """
    x = np.asarray(data_points, dtype=float)
    # unpack and enforce positivity for a,b,N0,c
    a, b, N0, c = params
    a = np.abs(a) + 1e-12
    b = np.abs(b) + 1e-12
    N0 = np.abs(N0) + 1e-12
    c = np.abs(c)
    pred = a * np.power(x + N0, -b) + c
    return pred

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (N, loss) data.

    Args:
      data_points: array-like of training data sizes
      loss_values: array-like of observed losses

    Returns:
      best_params: length-4 array [a, b, N0, c]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    # avoid zeros/negatives
    eps = 1e-12
    y = np.clip(y, eps, None)

    # parameter bounds: a in (0, max(y)*10), b in (0,10), N0 in (1e-8, max(N)*10), c in (0, max(y))
    bounds = [
        (1e-8, np.max(y)*10),    # a
        (1e-8, 10.0),            # b
        (1e-8, np.max(x)*10.0),  # N0
        (0.0,  np.max(y))        # c
    ]

    # Objective: mean squared error in log-space
    def objective_log_mse(params):
        pred = scaling_law_func(x, params)
        if np.any(pred <= 0):
            return np.inf
        return np.mean((np.log(pred + eps) - np.log(y))**2)

    # 1) Global search
    result_de = differential_evolution(
        objective_log_mse,
        bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=20,
        tol=1e-6,
        polish=False,
        seed=42
    )
    candidate = result_de.x

    # 2) Local refinement
    result_local = minimize(
        objective_log_mse,
        candidate,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-9, 'gtol':1e-8, 'maxiter':1000}
    )
    best_params = result_local.x if result_local.success else candidate

    return best_params

# Attach metadata
scaling_law_func.num_params = 4

# EVOLVE-BLOCK-END