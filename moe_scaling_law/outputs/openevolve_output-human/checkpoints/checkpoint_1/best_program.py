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
    
    # Enforce basic parameter constraints to avoid invalid regions
    E_start = max(E_start, 1.1)
    E_max = max(E_max, E_start + 1.0)
    
    # Compute Ê (E_hat)
    denom = E - 1.0 + (1.0 / E_start - 1.0 / E_max)
    denom = np.maximum(denom, 1e-8)
    one_over_E_hat = 1.0 / denom + 1.0 / E_max
    E_hat = 1.0 / np.maximum(one_over_E_hat, 1e-8)
    
    # Log terms
    log_N = np.log(N)
    log_E_hat = np.log(np.maximum(E_hat, 1e-8))
    
    # Scaling law in log-space
    log_loss = a * log_N + b * log_E_hat + c * log_N * log_E_hat + d
    
    return np.exp(log_loss)

# EVOLVE-BLOCK-START
def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the human-designed MoE scaling law to data using constrained L-BFGS-B optimization.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (6 parameters)
    """
    E = np.asarray(num_experts, dtype=float)
    N = np.asarray(total_parameter_count, dtype=float)
    L = np.asarray(loss_values, dtype=float)
    
    # Work in log-loss space for stability
    y_log = np.log(np.maximum(L, 1e-12))
    
    # Good initial guesses
    a0 = -0.5
    b0 = -0.5
    c0 = 0.1
    d0 = np.median(y_log)
    Es0 = np.clip(np.median(E), 1.1 + 1e-3, None)
    Em0 = np.clip(np.max(E), Es0 + 1.0 + 1e-3, None)
    initial_params = np.array([a0, b0, c0, d0, Es0, Em0], dtype=float)
    
    # Bounds: a,b,c,d unbounded, E_start >= 1.1, E_max >= E_start + 1
    bounds = [
        (None, None),  # a
        (None, None),  # b
        (None, None),  # c
        (None, None),  # d
        (1.1, None),   # E_start
        (1.1 + 1e-3, None)  # E_max (we'll enforce Em>Es+1 in objective corrections)
    ]
    
    def objective(params):
        # Enforce E_max > E_start + 1 within objective to keep valid region
        params = params.copy()
        if params[5] < params[4] + 1.0:
            params[5] = params[4] + 1.0 + 1e-6
        
        # Predicted losses and their logs
        pred = scaling_law_func(E, N, params)
        log_pred = np.log(np.maximum(pred, 1e-12))
        
        # MSE in log-space
        return np.mean((log_pred - y_log) ** 2)
    
    # Run optimization
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 1000,
            'ftol': 1e-12,
            'gtol': 1e-8
        }
    )
    
    # Return best parameters found
    if result.success:
        return result.x
    else:
        # Even if not fully converged, return the best found
        return result.x
# EVOLVE-BLOCK-END

# Annotate number of parameters
scaling_law_func.num_params = 6