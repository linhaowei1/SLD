# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models

Revised scaling law uses a sum of two power‐law regimes plus an offset. 
This compact 7‐parameter form can flexibly capture different scaling 
behaviors in both the expert count and dense‐parameter count dimensions.
    
    loss = a * E^{-b} * N^{-c}
         + d * E^{-e} * N^{-f}
         + g

where:
  - E is the number of experts (shifted slightly to avoid zeros)
  - N is the total parameter count (normalized to billions)
  - [a,b,c,d,e,f,g] are the learnable parameters
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Compute the predicted loss under a two‐regime power‐law model.

    Args:
        num_experts: Array of number of experts (E)
        total_parameter_count: Array of total parameter counts (N)
        params: Array of 7 parameters [a, b, c, d, e, f, g]

    Returns:
        Array of predicted losses
    """
    # Unpack the 7 parameters
    a, b, c, d, e, f, g = params

    # Safely cast and offset to avoid zero‐division or log issues
    E = num_experts.astype(float) + 1e-6
    N = total_parameter_count.astype(float)
    # Normalize N into billions for numerical stability
    N_norm = N / 1e9 + 1e-6

    # First power‐law regime
    term1 = a * np.power(E, -b) * np.power(N_norm, -c)
    # Second power‐law regime
    term2 = d * np.power(E, -e) * np.power(N_norm, -f)
    # Sum of regimes plus an offset
    loss_pred = term1 + term2 + g

    # Ensure non‐negative predictions
    return np.clip(loss_pred, 1e-12, np.inf)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the MoE scaling law parameters to data using BFGS.

    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of observed loss values

    Returns:
        Optimized parameter vector of length 7
    """
    # Initialize 7 parameters to 1
    initial_params = np.ones(7)

    def objective(params):
        try:
            preds = scaling_law_func(num_experts, total_parameter_count, params)
            return np.mean((preds - loss_values) ** 2)
        except:
            # Large penalty for invalid parameter regions
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# Declare how many parameters the scaling law uses
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END