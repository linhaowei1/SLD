"""
Human-designed MoE Scaling Law
log L(N,E) = a*log N + b*log Ê + c*log N * log Ê + d
where 1/Ê = 1/(E-1+(1/E_start - 1/E_max)) + 1/E_max
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Human-designed MoE scaling law function.
    
    log L(N,E) = a*log N + b*log Ê + c*log N * log Ê + d
    where 1/Ê = 1/(E-1+(1/E_start - 1/E_max)) + 1/E_max
    
    Simplified version using 6 parameters: [a, b, c, d, E_start, E_max]
    
    Args:
        num_experts: Array of number of experts (E)
        total_parameter_count: Array of total parameter counts (N)
        params: Array of parameters [a, b, c, d, E_start, E_max] (6 parameters)
        
    Returns:
        Predicted loss values
    """
    a, b, c, d, E_start, E_max = params
    
    # Convert to numpy arrays and ensure positive values
    E = np.asarray(num_experts, dtype=float) + 1e-8
    N = np.asarray(total_parameter_count, dtype=float) + 1e-8
    
    # Calculate Ê (E_hat)
    # 1/Ê = 1/(E-1+(1/E_start - 1/E_max)) + 1/E_max
    E_start = max(E_start, 1.1)  # Ensure E_start > 1
    E_max = max(E_max, E_start + 1)  # Ensure E_max > E_start
    
    denominator = E - 1 + (1/E_start - 1/E_max)
    denominator = np.maximum(denominator, 1e-8)  # Avoid division by zero
    
    one_over_E_hat = 1.0 / denominator + 1.0 / E_max
    E_hat = 1.0 / np.maximum(one_over_E_hat, 1e-8)
    
    # Calculate log terms
    log_N = np.log(N)
    log_E_hat = np.log(np.maximum(E_hat, 1e-8))
    
    # Apply the scaling law: log L = a*log N + b*log Ê + c*log N * log Ê + d
    log_loss = a * log_N + b * log_E_hat + c * log_N * log_E_hat + d
    
    # Convert back to loss (exp of log_loss)
    loss = np.exp(log_loss)
    
    return loss

# EVOLVE-BLOCK-START
def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the human-designed MoE scaling law to data using BFGS optimization.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (6 parameters)
    """
    # Initialize parameters: [a, b, c, d, E_start, E_max]
    initial_params = np.array([-0.5, -0.5, 0.1, 0.0, 2.0, 100.0])
    
    def objective(params):
        try:
            predicted = scaling_law_func(num_experts, total_parameter_count, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6
    
    result = minimize(objective, initial_params, method='BFGS')
    
    final_params = result.x if result.success else initial_params
    
    return final_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 6