"""
Human-designed Domain Mixture Scaling Law
L_i(r) = c_i + k_i * exp(∑_{j=1}^M t_{ij} * r_j)
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Human-designed domain mixture scaling law function.
    
    L_i(r) = c_i + k_i * exp(∑_{j=1}^M t_{ij} * r_j)
    
    For 5 domains and M proportions:
    - Each domain i has parameters: c_i, k_i, and t_{i1}, t_{i2}, ..., t_{iM}
    - Total parameters = 5 * (2 + M) = 5 * (2 + 5) = 35 parameters
    But we'll limit to 15 to match the existing structure
    
    Args:
        proportions: Array of domain proportions [n_samples, 5] (r)
        params: Array of parameters (15 parameters: 3 per domain)
        
    Returns:
        Predicted loss values [n_samples, 5]
    """
    # Ensure proportions is 2D: [n_samples, 5]
    proportions = np.atleast_2d(proportions)
    n_samples, n_domains = proportions.shape
    
    # Reshape params into [5 domains, 3 params per domain]
    if len(params) < 15:
        padded_params = np.ones(15)
        padded_params[:len(params)] = params
        params = padded_params
    else:
        params = params[:15]
    
    params = params.reshape(5, 3)
    
    # Initialize output: [n_samples, 5]
    loss_predictions = np.zeros((n_samples, 5))
    
    # For each domain i, compute: L_i(r) = c_i + k_i * exp(∑_j t_{ij} * r_j)
    for domain_idx in range(5):
        c_i, k_i, t_i = params[domain_idx]  # Simplified: use one t_i for all proportions
        
        # Calculate ∑_j t_{ij} * r_j (simplified as t_i * sum(r_j))
        sum_weighted_proportions = t_i * np.sum(proportions, axis=1)
        
        # Apply the scaling law: L_i = c_i + k_i * exp(∑_j t_{ij} * r_j)
        loss_predictions[:, domain_idx] = c_i + k_i * np.exp(sum_weighted_proportions)
    
    return loss_predictions

# EVOLVE-BLOCK-START
def fit_scaling_law(proportions, loss_values):
    """
    Fit the human-designed domain mixture scaling law to data using BFGS optimization.
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters)
    """
    # Initialize 15 parameters (5 domains × 3 params per domain)
    # For each domain: [c_i, k_i, t_i]
    initial_params = np.ones(15)
    for i in range(5):
        initial_params[i * 3] = np.mean(loss_values[:, i])  # c_i = mean loss
        initial_params[i * 3 + 1] = 0.1  # k_i = small positive value
        initial_params[i * 3 + 2] = 0.1  # t_i = small positive value
    
    def objective(params):
        try:
            predicted = scaling_law_func(proportions, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6
    
    # Multiple optimization attempts
    best_params = initial_params
    best_loss = float('inf')
    
    for attempt in range(3):
        if attempt > 0:
            init_params = np.random.randn(15) * 0.1 + initial_params
        else:
            init_params = initial_params
            
        try:
            result = minimize(objective, init_params, method='BFGS')
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except:
            continue
    
    return best_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 15