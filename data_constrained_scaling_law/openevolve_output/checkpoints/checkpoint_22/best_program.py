# EVOLVE-BLOCK-START
"""
Data‐constrained scaling law discovery for LLM training scenarios.
Improved functional form and robust fitting for up to 7 parameters.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given tokens, model size, and available unique tokens.
    
    Functional form:
      ratio = tokens / unique_tokens
      rep_factor = (1 - exp(- ratio / d))^g
      Teff = unique_tokens * rep_factor
      loss = a + b * model_size^{-p} + c * Teff^{-q}
    
    params = [a, b, p, c, q, d, g]  (7 parameters)
    """
    # Unpack parameters
    a, b, p, c, q, d, g = params
    # Prevent division by zero
    U = np.maximum(unique_tokens, 1.0)
    # Repetition ratio
    ratio = tokens / U
    # Effective data coverage factor
    rep_factor = np.power(1.0 - np.exp(- ratio / np.maximum(d, 1e-8)), np.maximum(g, 1e-8))
    Teff = U * rep_factor
    # Model‐size and data contributions
    loss = a + b * np.power(model_size, -np.maximum(p, 1e-8)) \
               + c * np.power(Teff + 1e-8, -np.maximum(q, 1e-8))
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law to observed loss values.
    
    Uses bounded nonlinear least squares for stability.
    """
    # Initial parameter guesses
    a0 = np.min(loss_values) * 0.9
    b0 = (np.max(loss_values) - np.min(loss_values)) * 0.5
    c0 = b0
    p0, q0 = 0.5, 0.5
    d0, g0 = 1.0, 1.0
    x0 = np.array([a0, b0, p0, c0, q0, d0, g0])

    # Parameter bounds
    lower = np.array([0.0,    0.0,    1e-3,   0.0,    1e-3,   1e-6,  1e-3])
    upper = np.array([np.inf, np.inf, 10.0,   np.inf, 10.0,   np.inf, 10.0])

    def residuals(params):
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        return pred - loss_values

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method='trf',
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=2000
        )
        params_opt = res.x if res.success else x0
    except Exception:
        params_opt = x0

    return params_opt

# Expose expected number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END