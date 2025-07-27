# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models.
Enhanced scaling law form with 6 parameters:
    loss = a 
           + b / (E^c) 
           + d / (N^e) 
           + f * log1p(E) / log1p(N)
This captures power-law decay in experts and parameters,
plus a saturating interaction term via logarithms.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A scaling law function to model how loss scales with number of experts (E)
    and total parameter count (N) in Mixture-of-Experts models.

    Args:
        num_experts:        Array-like of expert counts (E)
        total_parameter_count: Array-like of total parameter counts (N)
        params:             Length-6 array of parameters [a, b, c, d, e, f]

    Returns:
        Array of predicted loss values.
    """
    # ensure numeric arrays
    E = np.array(num_experts, dtype=float)
    N = np.array(total_parameter_count, dtype=float)

    # unpack parameters
    a, b, c, d, e, f = params

    # term for expert scaling: b / E^c
    term_E = b / (np.power(E + 1e-6, c) + 1e-12)
    # term for parameter scaling: d / N^e
    term_N = d / (np.power(N + 1e-6, e) + 1e-12)
    # interaction term: f * log(1+E) / log(1+N)
    term_int = f * np.log1p(E) / (np.log1p(N) + 1e-12)

    # base offset + contributions
    loss = a + term_E + term_N + term_int
    return loss

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the MoE scaling law to data using BFGS.

    Args:
        num_experts:           Array-like of expert counts
        total_parameter_count: Array-like of total parameter counts
        loss_values:           Array-like of observed loss values

    Returns:
        Optimized params (length 6).
    """
    # initialize all 6 parameters to 1
    initial_params = np.ones(6)

    def objective(params):
        try:
            preds = scaling_law_func(num_experts, total_parameter_count, params)
            return np.mean((preds - loss_values) ** 2)
        except:
            return 1e6  # large penalty on failure

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# declare number of parameters for downstream code
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END