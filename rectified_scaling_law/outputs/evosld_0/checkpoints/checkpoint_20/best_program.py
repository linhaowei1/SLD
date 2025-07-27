# EVOLVE-BLOCK-START
"""
Refined scaling law discovery for LLM finetuning scenarios.

We model loss as a 4-parameter function:
    L(N) = a * (N + N0)^(-b) + c

and fit it by minimizing the normalized mean squared error (NMSE)
using a hybrid global (Differential Evolution) + local (L-BFGS-B) search.
This approach yields robust, generalizable fits across diverse datasets
and model sizes, with numeric stability and parameter efficiency.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law:
        L(N) = a * (N + N0)^(-b) + c

    Args:
        data_points: array-like of training data sizes (N)
        params: array-like [a, b, c, N0]
            a (>0): scale coefficient
            b (>0): power-law exponent
            c (>=0): asymptotic minimum loss
            N0 (>=0): data-size offset for small-N behavior
    Returns:
        numpy array of predicted losses
    """
    a, b, c, N0 = params
    x = np.asarray(data_points, dtype=np.float64)
    # ensure non-negative effective data size
    x_eff = x + max(N0, 0.0)
    return a * np.power(x_eff, -b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to observed (N, loss) pairs
    by minimizing normalized MSE (NMSE):
        NMSE = MSE(pred, obs) / Var(obs)

    Uses a two-phase optimization:
      1) Global search via Differential Evolution
      2) Local refinement via L-BFGS-B

    Args:
        data_points: array-like of training data sizes
        loss_values: array-like of observed losses
    Returns:
        numpy array of best-fit parameters [a, b, c, N0]
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # safe defaults
    N_max = max(x.max(), 1.0)
    y_max = max(y.max(), 1e-8)
    y_var = np.var(y) if np.var(y) > 0 else 1.0

    # parameter bounds
    bounds = [
        (1e-8, y_max * 10.0),  # a
        (1e-8, 10.0),          # b
        (0.0, y_max * 1.0),    # c
        (0.0, N_max)           # N0
    ]

    # objective: normalized MSE
    def nmse_obj(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2) / y_var

    # --- 1) Global search: Differential Evolution ---
    de_result = differential_evolution(
        nmse_obj,
        bounds,
        strategy='best1bin',
        maxiter=800,
        popsize=15,
        tol=1e-6,
        polish=False,
        disp=False,
        seed=42
    )
    init_params = de_result.x

    # --- 2) Local refinement: L-BFGS-B ---
    local_result = minimize(
        nmse_obj,
        init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'maxiter': 2000}
    )

    best_params = local_result.x if local_result.success else init_params
    return best_params

# metadata: number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END