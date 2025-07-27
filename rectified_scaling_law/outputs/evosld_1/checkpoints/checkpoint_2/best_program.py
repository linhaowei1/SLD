# EVOLVE-BLOCK-START
"""
Enhanced scaling law discovery for LLM finetuning scenarios.
We model loss vs. data size with a 4-parameter power-law + offset:
    loss(x) = a * (x + x0)^(-b) + c
and fit parameters [a, b, x0, c] via a global+local optimization
with bounds and robust initialization.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

def scaling_law_func(data_points, params):
    """
    Scaling law: loss = a * (x + x0)^(-b) + c

    Params (4):
        a  : scale coefficient (>0)
        b  : exponent (>0)
        x0 : horizontal shift (>=0)
        c  : asymptotic loss floor (>=0)
    """
    x = np.asarray(data_points, dtype=np.float64)
    a, b, x0, c = params
    # enforce non-negativity for stability
    a = np.abs(a)
    b = np.abs(b)
    x0 = np.abs(x0)
    c = np.abs(c)
    # avoid 0^(-b) by ensuring x + x0 > 0
    shifted = x + x0
    shifted = np.where(shifted > 0, shifted, 1e-8)
    return a * shifted**(-b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (data_points, loss_values)
    using a hybrid global (differential evolution) + local (L-BFGS-B) search.
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # Define reasonable parameter bounds
    bounds = [
        (1e-8, np.max(y)*10 + 1.0),       # a: scale ~ loss range
        (1e-5, 10.0),                     # b: exponent
        (0.0, np.max(x)),                 # x0: shift up to max data size
        (0.0, np.max(y)*2 + 1.0)          # c: baseline up to ~2x loss range
    ]

    # Objective: MSE between predicted and actual losses
    def objective(params):
        pred = scaling_law_func(x, params)
        return np.mean((pred - y) ** 2)

    # 1) Global search with Differential Evolution
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=False,
            disp=False
        )
        init_params = de_result.x
    except Exception:
        # Fallback initialization
        init_params = np.array([
            np.ptp(y),            # a ~ range of y
            0.5,                  # b ~ moderate exponent
            np.median(x),         # x0 ~ median data size
            np.min(y)             # c ~ min observed loss
        ])

    # 2) Local refinement with L-BFGS-B
    try:
        local_result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if local_result.success:
            fitted = local_result.x
        else:
            fitted = init_params
    except Exception:
        fitted = init_params

    return fitted

# Number of parameters exposed for external checks (must be ≤ 4)
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END