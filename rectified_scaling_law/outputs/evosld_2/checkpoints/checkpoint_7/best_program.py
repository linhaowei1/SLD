# EVOLVE-BLOCK-START
"""
Enhanced scaling law discovery for LLM finetuning scenarios.

We fit a 4‐parameter form:
    L(N) = A * (N + N0)^(-α) + B

Parameters:
  - A  : scale of the power‐law term
  - α  : power‐law exponent
  - B  : irreducible loss floor
  - N0 : horizontal shift to handle small-N behaviour

Features:
  - 4 parameters (A, α, B, N0) for expressiveness without overfitting
  - Bounds to ensure numerical stability and physically meaningful parameters
  - Multiple random restarts with L‐BFGS‐B for robust convergence
  - MSE objective (minimized) correlates with NMSE evaluation
  - Simple, maintainable, and theoretically grounded form
"""

import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict loss given training data size(s) using a 4‐parameter scaling law.
    
    L(N) = A * (N + N0)^(-α) + B
    
    Inputs:
      data_points : array‐like of N (training sample sizes)
      params       : array‐like of 4 parameters [A, α, B, N0]
    Returns:
      Array of predicted losses.
    """
    A, alpha, B, N0 = params
    x = np.asarray(data_points, dtype=float)
    # Ensure positivity inside power
    shifted = x + np.abs(N0)
    # Compute safely
    with np.errstate(divide='ignore', invalid='ignore'):
        y = A * np.power(shifted, -alpha) + B
    # Replace any nan/inf by large values to penalize in optimization
    y = np.where(np.isfinite(y), y, np.finfo(float).max)
    return y

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4‐parameter scaling law to (data_points, loss_values).
    Uses multiple L‐BFGS‐B restarts to avoid bad local minima.
    
    Returns:
      best_params : array of 4 optimized parameters [A, α, B, N0]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    # Basic statistics for bounds and initialization
    N_max, N_min = x.max(), x.min()
    y_max, y_min = y.max(), y.min()
    
    # Parameter bounds: A>0, α>0, B>=0, N0>=0
    bounds = [
        (1e-8, max(y_max * 10, 1e-2)),  # A
        (1e-8, 5.0),                    # α
        (0.0, max(y_max * 2, 1.0)),     # B
        (0.0, N_max)                    # N0
    ]
    
    # Objective: mean squared error
    def objective(p):
        y_pred = scaling_law_func(x, p)
        return np.mean((y_pred - y) ** 2)
    
    # Generate several random inits within bounds
    rng = np.random.RandomState(42)
    inits = []
    # a) heuristic init
    A0 = (y_max - y_min) * (N_max ** 0.5)
    alpha0 = 0.5
    B0 = max(y_min * 0.9, 0.0)
    N00 = 0.0
    inits.append([A0, alpha0, B0, N00])
    # b) a few random
    for _ in range(4):
        init = [
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1]),
            rng.uniform(bounds[3][0], bounds[3][1])
        ]
        inits.append(init)
    
    best_cost = np.inf
    best_params = None
    
    for init in inits:
        res = minimize(
            objective,
            x0=init,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res.success and res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x.copy()
    
    # Fallback to heuristic if optimization fails
    if best_params is None:
        best_params = np.array(inits[0], dtype=float)
    
    return best_params

# Specify that scaling_law_func expects 4 parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END