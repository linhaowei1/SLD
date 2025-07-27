# EVOLVE-BLOCK-START
"""
Enhanced MoE scaling law discovery for Mixture of Experts models.
Now uses a multiplicative power-law form with shift parameters and an additive offset,
totaling 6 parameters for improved flexibility and stability.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict validation loss given number of experts and parameter count.
    
    Model: L = A * (N + N0)^(-alpha) * (P + P0)^(-beta) + C
    where:
      N  = num_experts
      P  = total_parameter_count
      params = [A, N0, P0, alpha, beta, C]
    
    Returns:
      Array of predicted losses.
    """
    A, N0, P0, alpha, beta, C = params
    # Shift and ensure positivity
    ne = np.maximum(num_experts + N0, 1e-8)
    pt = np.maximum(total_parameter_count + P0, 1e-8)
    # Multiplicative power-law plus offset
    return A * (ne ** (-alpha)) * (pt ** (-beta)) + C

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter scaling law to data via bounded L-BFGS-B.
    
    Args:
      num_experts: array-like of expert counts
      total_parameter_count: array-like of parameter counts
      loss_values: array-like of observed losses
      
    Returns:
      params: Optimized array [A, N0, P0, alpha, beta, C]
    """
    # Initial parameter guesses
    A0 = max(loss_values.max() - loss_values.min(), 1e-1)
    C0 = max(loss_values.min(), 1e-3)
    N0_0 = 1.0
    P0_0 = np.median(total_parameter_count) * 0.1
    alpha0 = 0.5
    beta0 = 0.5
    x0 = np.array([A0, N0_0, P0_0, alpha0, beta0, C0], dtype=float)

    # Bounds to ensure positive scaling and shifts
    bounds = [
        (1e-8, None),   # A > 0
        (0.0, None),    # N0 >= 0
        (0.0, None),    # P0 >= 0
        (1e-8, 10.0),   # alpha > 0
        (1e-8, 10.0),   # beta > 0
        (None, None)    # C unconstrained
    ]

    # Objective: mean squared error
    def objective(x):
        pred = scaling_law_func(num_experts, total_parameter_count, x)
        return np.mean((pred - loss_values) ** 2)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-12}
    )

    if result.success:
        return result.x
    else:
        # Fallback to initial guess on failure
        return x0

# Inform downstream code of parameter count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END