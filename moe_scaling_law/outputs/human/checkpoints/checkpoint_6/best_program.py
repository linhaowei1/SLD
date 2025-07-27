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
    
    denominator = E - 1 + 1/(1/E_start - 1/E_max)
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
    Fit the human-designed MoE scaling law to data using a two-stage
    parameterization and L-BFGS-B optimization in log-loss space.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters [a, b, c, d, E_start, E_max]
    """
    # Convert inputs to numpy arrays
    E = np.asarray(num_experts, dtype=float)
    N = np.asarray(total_parameter_count, dtype=float)
    L = np.asarray(loss_values, dtype=float)
    
    # Work in log-loss space for stability
    eps = 1e-8
    logL = np.log(L + eps)
    
    # Parameterize E_start > 1 and E_max > E_start via logs:
    #   E_start = 1 + exp(p4)
    #   E_max = E_start + exp(p5)
    # Unconstrained parameters: [a, b, c, d, p4, p5]
    
    # Initial guesses
    a0 = -0.5
    b0 = -0.5
    c0 =  0.1
    d0 = np.median(logL)        # intercept roughly median log-loss
    p4_0 = np.log(1.0)          # initial E_start = 1 + 1 = 2
    p5_0 = np.log(max(E.max() - 1.0, 1.0))  # initial E_max ~ E.max()
    x0 = np.array([a0, b0, c0, d0, p4_0, p5_0])
    
    # Transform unconstrained -> actual params
    def unpack(x):
        a, b, c, d, p4, p5 = x
        E_start = 1.0 + np.exp(p4)
        E_max   = E_start + np.exp(p5)
        return np.array([a, b, c, d, E_start, E_max], dtype=float)
    
    # Objective: MSE in log-loss space
    def obj(x):
        params = unpack(x)
        # Predict loss via scaling law
        pred = scaling_law_func(E, N, params)
        # Numerical safeguards
        if np.any(pred <= 0) or not np.isfinite(pred).all():
            return 1e6
        log_pred = np.log(pred + eps)
        return np.mean((log_pred - logL) ** 2)
    
    # Run L-BFGS-B optimization (unconstrained since paramization enforces positivity)
    res = minimize(
        obj, x0, method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Extract final parameters
    x_opt = res.x if res.success else x0
    final_params = unpack(x_opt)
    return final_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 6