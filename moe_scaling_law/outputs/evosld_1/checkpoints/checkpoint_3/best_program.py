# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models
Evolved version: incorporates a cross-term and uses constrained optimization
for stability and improved fit.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A 6-parameter scaling law for MoE models that captures
    individual and joint effects of experts and parameter count.

    loss = A
         + B / (ne^alpha)
         + C / (np^beta)
         + D / (ne^alpha * np^beta)

    where ne = num_experts + eps_n
          np = (total_parameter_count / 1e6) + eps_p

    Args:
        num_experts: array-like, number of experts
        total_parameter_count: array-like, total model parameter count
        params: array-like of length 6:
            [A, B, alpha, C, beta, D]

    Returns:
        loss_pred: array-like, predicted loss values
    """
    # Unpack parameters
    A, B, alpha, C, beta, D = params

    # Convert inputs to floats and add small eps to avoid zero divisions
    ne = np.asarray(num_experts, dtype=np.float64) + 1e-6
    # scale parameters to millions to keep exponents in a reasonable range
    np_par = np.asarray(total_parameter_count, dtype=np.float64) / 1e6 + 1e-6

    # Compute power-law terms
    ne_term = np.power(ne, alpha)
    np_term = np.power(np_par, beta)

    # Compose loss prediction
    loss_pred = (
        A
        + B / ne_term
        + C / np_term
        + D / (ne_term * np_term)
    )

    return loss_pred

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to data using bounded L-BFGS-B.

    Args:
        num_experts: array-like, number of experts
        total_parameter_count: array-like, total model parameter count
        loss_values: array-like, ground-truth loss values

    Returns:
        params_opt: array of length 6 with optimized parameters
    """
    # Initial guesses: A ~ min(loss), B,C,D ~ half range, exponents ~ 0.5
    lv = np.asarray(loss_values, dtype=np.float64)
    A0 = max(np.min(lv) * 0.9, 1e-3)
    span = (np.max(lv) - np.min(lv) + 1e-3)
    initial = np.array([
        A0,        # A
        span * 0.5,  # B
        0.5,       # alpha
        span * 0.3,  # C
        0.5,       # beta
        span * 0.2   # D
    ], dtype=np.float64)

    # Bounds: A >= 0, B,C,D >= 0, exponents in [1e-6, 5]
    bounds = [
        (0.0, None),      # A
        (1e-8, None),     # B
        (1e-6, 5.0),      # alpha
        (1e-8, None),     # C
        (1e-6, 5.0),      # beta
        (1e-8, None)      # D
    ]

    # Objective: mean squared error
    def objective(p):
        pred = scaling_law_func(num_experts, total_parameter_count, p)
        # guard against NaNs or infs
        if not np.all(np.isfinite(pred)):
            return 1e6 + np.sum(~np.isfinite(pred)) * 1e3
        return np.mean((pred - lv) ** 2)

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-12}
    )

    if result.success and len(result.x) == 6:
        params_opt = result.x
    else:
        # fallback to initial if optimization fails
        params_opt = initial

    return params_opt

# Declare how many parameters the scaling law expects
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END