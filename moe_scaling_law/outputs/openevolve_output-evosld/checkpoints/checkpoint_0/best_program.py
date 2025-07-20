# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models

This implementation uses a 6-parameter power-law form with an
interaction term between number of experts and total parameter count.
Optimization is performed with bounds to ensure numerical stability.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict validation loss given MoE configuration.

    Implements:
        loss = c0 
               + A / (E^alpha) 
               + B / (P^beta) 
               + C / (E^alpha * P^beta)
    where
        E = num_experts + eps_e
        P = total_parameter_count + eps_p

    Args:
        num_experts: array-like, number of experts (E)
        total_parameter_count: array-like, total parameters (P)
        params: 6-element array [c0, A, B, C, alpha, beta]

    Returns:
        pred_loss: array-like, predicted losses
    """
    # Unpack parameters
    c0, A, B, C, alpha, beta = params

    # Small epsilons to avoid zeros / divisions
    eps_e = 1e-6
    eps_p = 1e-6

    E = np.maximum(num_experts, 0.0) + eps_e
    P = np.maximum(total_parameter_count, 0.0) + eps_p

    # Compute power-law terms
    term_e = E ** (-alpha)
    term_p = P ** (-beta)

    # Final prediction
    pred_loss = c0 + A * term_e + B * term_p + C * (term_e * term_p)
    return pred_loss

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to data by minimizing MSE.

    Args:
        num_experts: array-like, number of experts
        total_parameter_count: array-like, total parameters
        loss_values: array-like, observed validation losses

    Returns:
        params_opt: np.ndarray of length 6, fitted parameters
    """
    # Ensure numpy arrays
    E = np.asarray(num_experts, dtype=np.float64)
    P = np.asarray(total_parameter_count, dtype=np.float64)
    L = np.asarray(loss_values, dtype=np.float64)

    # Initial guess: [c0, A, B, C, alpha, beta]
    init = np.array([0.5, 1.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float64)

    # Bounds to keep parameters in stable, interpretable ranges
    # c0, A, B, C >= 0; alpha, beta >= 0
    bnds = [
        (0.0, None),   # c0
        (0.0, None),   # A
        (0.0, None),   # B
        (0.0, None),   # C
        (0.0, 10.0),   # alpha
        (0.0, 10.0),   # beta
    ]

    def mse_obj(params):
        # Predict and compute mean squared error
        pred = scaling_law_func(E, P, params)
        # If invalid values appear, return large penalty
        if not np.all(np.isfinite(pred)):
            return 1e6
        return np.mean((pred - L) ** 2)

    # Optimize with L-BFGS-B within bounds
    result = minimize(
        mse_obj,
        init,
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': 10000, 'ftol': 1e-12}
    )

    if result.success:
        params_opt = result.x
    else:
        # Fallback to initial guess on failure
        params_opt = init

    return params_opt

# Specify number of parameters used
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END