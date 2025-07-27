# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models
Evolved program with an enhanced scaling law form combining 
multiplicative decay and logarithmic correction terms.
Supports up to 6 parameters.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Enhanced MoE scaling law function:
      loss ≈ p0 * (E+1)^(-p1) * (N+1)^(-p2)
             + p3 * log(E+1)
             + p4 * log(N+1)
             + p5

    Args:
        num_experts:           Array of expert counts (E)
        total_parameter_count: Array of total parameter counts (N)
        params:                Array [p0, p1, p2, p3, p4, p5]
    Returns:
        Predicted loss array
    """
    p0, p1, p2, p3, p4, p5 = params
    # Multiplicative decay term
    decay = p0 * np.power(num_experts + 1, -p1) * np.power(total_parameter_count + 1, -p2)
    # Logarithmic correction terms
    log_e = p3 * np.log(num_experts + 1)
    log_n = p4 * np.log(total_parameter_count + 1)
    # Combine with a constant offset
    return decay + log_e + log_n + p5

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to data using BFGS.
    """
    initial_params = np.ones(6)

    def objective(params):
        try:
            pred = scaling_law_func(num_experts, total_parameter_count, params)
            return np.mean((pred - loss_values) ** 2)
        except:
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# Inform downstream code how many params this scaling law uses
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END