# EVOLVE-BLOCK-START
"""
Improved data-constrained scaling law for LLM loss:
  loss = p0 
       + p1 * (M/M_max)^(-p2)
       + p3 * (T/T_max)^(-p4)
       + p5 * (1 - exp(-p6 * (T/T_max)/(U/U_max)))

7 parameters: [p0,p1,p2,p3,p4,p5,p6]
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given tokens, model size, unique tokens and params.
    Args:
        tokens        : array-like, training tokens used
        model_size    : array-like, model parameter counts
        unique_tokens : array-like, unique tokens available
        params        : length-7 array [p0,p1,p2,p3,p4,p5,p6]
    Returns:
        loss_pred     : array of predicted losses
    """
    p0, p1, p2, p3, p4, p5, p6 = params

    # scale features to [0,1] for stability
    M = model_size / (np.max(model_size) + 1e-12)
    T = tokens     / (np.max(tokens)     + 1e-12)
    U = unique_tokens / (np.max(unique_tokens) + 1e-12)

    dup = T / (U + 1e-12)
    # saturating duplication effect
    dup_effect = 1.0 - np.exp(-p6 * dup)

    loss = p0 \
           + p1 * M**(-p2) \
           + p3 * T**(-p4) \
           + p5 * dup_effect

    return loss


def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law to data by minimizing MSE.
    Returns optimized params array [p0..p6].
    """
    # to numpy
    T = np.asarray(tokens, dtype=float)
    M = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # determine sensible bounds from data
    y_max = np.max(y)
    bounds = [
        (0.0,        y_max),    # p0: baseline loss floor
        (1e-8,       y_max),    # p1: model-size scale
        (1e-3,       5.0),      # p2: model-size exponent
        (1e-8,       y_max),    # p3: training-tokens scale
        (1e-3,       5.0),      # p4: training-tokens exponent
        (1e-8,       y_max),    # p5: duplication asymptotic gain
        (1e-3,       5.0)       # p6: duplication rate exponent
    ]

    # objective: mean squared error
    def _obj(p):
        pred = scaling_law_func(T, M, U, p)
        return np.mean((pred - y)**2)

    # multi-start L-BFGS-B
    best_p = None
    best_val = np.inf
    rng = np.random.RandomState(0)
    n_restarts = 10
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(
            _obj,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_p = res.x

    # fallback to midpoints if all fail
    if best_p is None:
        best_p = np.array([(lo + hi) / 2.0 for lo, hi in bounds])

    return best_p

# metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END