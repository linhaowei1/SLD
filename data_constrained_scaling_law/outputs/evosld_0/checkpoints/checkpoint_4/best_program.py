# EVOLVE-BLOCK-START
"""
Evolved data-constrained scaling law for LLM training loss,
incorporating data saturation via unique tokens and robust fitting.
Uses at most 7 parameters (here 6).
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given training tokens, model size, and unique-token budget.

    params = [a, b, c, alpha, beta, k]
      a     : asymptotic minimum loss (L_inf)
      b     : model-size coefficient
      c     : data-effective coefficient
      alpha : model-size exponent (>0)
      beta  : effective-data exponent (>0)
      k     : data-saturation factor (>0)

    Effective data seen E = U * (1 - exp(-T / (k * U))),
    so that repeated passes over U saturate.

    Returns:
        loss: Array of same shape as inputs
    """
    a, b, c, alpha, beta, k = params

    # Prevent non-positive and scale down for numerical stability
    T = np.maximum(tokens, 0.0) / 1e7
    N = np.maximum(model_size, 1.0) / 1e7
    U = np.maximum(unique_tokens, 1.0) / 1e7
    k = max(k, 1e-8)

    # Effective data with saturation
    E = U * (1.0 - np.exp(-T / (k * U)))

    # Model-size and data-size contributions
    loss = a + b * N ** (-alpha) + c * E ** (-beta)

    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the scaling law parameters to observed losses.

    Returns:
        params: Array of 6 fitted parameters [a,b,c,alpha,beta,k]
    """
    # Initial guesses based on data
    Lmin, Lmax = float(np.min(loss_values)), float(np.max(loss_values))
    initial = np.array([
        Lmin,                # a
        (Lmax - Lmin) * 0.5, # b
        (Lmax - Lmin) * 0.5, # c
        0.3,                 # alpha
        0.3,                 # beta
        1.0                  # k
    ], dtype=float)

    # Parameter bounds for stability
    bounds = [
        (0.0, Lmax),        # a >= 0
        (1e-8, 10*(Lmax-Lmin)+1e-6),  # b > 0
        (1e-8, 10*(Lmax-Lmin)+1e-6),  # c > 0
        (1e-3, 2.0),        # alpha in (0,2]
        (1e-3, 2.0),        # beta in (0,2]
        (1e-3, 10.0)        # k positive
    ]

    # Objective: mean squared error in log-space for balanced fit
    def objective(p):
        pred = scaling_law_func(tokens, model_size, unique_tokens, p)
        # avoid log of zero
        lp = np.log(np.maximum(pred, 1e-8))
        lt = np.log(np.maximum(loss_values, 1e-8))
        return np.mean((lp - lt) ** 2)

    result = minimize(
        objective,
        x0=initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-9}
    )

    if result.success:
        fitted = result.x
    else:
        fitted = initial

    # Return at most 7 params (we use 6 here)
    return fitted

# Number of parameters expected
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END