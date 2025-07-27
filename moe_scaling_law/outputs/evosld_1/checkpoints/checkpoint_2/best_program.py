# EVOLVE-BLOCK-START
"""
Evolved MoE scaling law discovery for Mixture-of-Experts models.
This version uses a 6-parameter form that captures joint and marginal
effects of total parameter count and number of experts on validation loss.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Mixture-of-Experts scaling law:
      L(E, N) = k0 
               + k1 * N^{-alpha} * E^{-beta}
               + k2 * N^{-alpha}
               + k3 * E^{-beta}
    where:
      - E = num_experts
      - N = total_parameter_count
      - params = [k0, k1, k2, k3, alpha, beta] (6 parameters)

    Args:
        num_experts: array-like, number of experts (E)
        total_parameter_count: array-like, total (dense) parameter counts (N)
        params: length-6 array of parameters [k0, k1, k2, k3, alpha, beta]

    Returns:
        loss_pred: array of predicted losses
    """
    k0, k1, k2, k3, alpha, beta = params
    E = np.asarray(num_experts, dtype=float) + 1e-6
    N = np.asarray(total_parameter_count, dtype=float) + 1e-6

    invN_alpha = np.power(N, -alpha)
    invE_beta  = np.power(E, -beta)

    loss_pred = (
        k0
        + k1 * invN_alpha * invE_beta
        + k2 * invN_alpha
        + k3 * invE_beta
    )
    return loss_pred

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law by minimizing mean squared error.

    Args:
        num_experts: array-like, number of experts per experiment
        total_parameter_count: array-like, total dense parameter counts
        loss_values: array-like, observed validation loss values

    Returns:
        params_opt: optimized parameter array [k0, k1, k2, k3, alpha, beta]
    """
    # Convert inputs to numpy arrays
    E = np.asarray(num_experts, dtype=float)
    N = np.asarray(total_parameter_count, dtype=float)
    L = np.asarray(loss_values, dtype=float)

    # Initialize parameters:
    #   k0 around min observed loss,
    #   other ks ~1.0, alphas ~0.5
    init_params = np.array([
        np.clip(np.min(L), 0.0, None),
        1.0,
        1.0,
        1.0,
        0.5,
        0.5
    ], dtype=float)

    # Bounds to ensure positivity where needed
    bounds = [
        (0.0, None),   # k0
        (0.0, None),   # k1
        (0.0, None),   # k2
        (0.0, None),   # k3
        (0.0, 5.0),    # alpha
        (0.0, 5.0)     # beta
    ]

    # Objective: mean squared error between predicted and observed loss
    def objective(params):
        preds = scaling_law_func(E, N, params)
        return np.mean((preds - L) ** 2)

    result = minimize(
        objective,
        x0=init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )

    if result.success:
        return result.x
    else:
        # Fallback to initial params if optimization fails
        return init_params

# Attach metadata: number of parameters expected by scaling_law_func
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END