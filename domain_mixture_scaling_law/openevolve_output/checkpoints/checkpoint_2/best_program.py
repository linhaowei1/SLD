# EVOLVE-BLOCK-START
"""
Domain mixture scaling law discovery for LLM training scenarios
Initial program with a simple power law form that treats each proportion equally
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    A scaling law function to model the relationship between domain proportions and loss for each domain.
    
    This models 5 domain loss equations using a simple power law:
    Li = A_i + B_i * (∑_j proportion_ij^C_i)
    
    All domains use the same mathematical expression but different coefficients.
    Each domain uses 3 parameters: A_i (bias), B_i (coefficient), C_i (power)
    Total: 15 parameters for 5 domains
    
    Args:
        proportions: Array of domain proportions [n_samples, 5] where each row sums to 1.0
        params: Array of parameters for the scaling law (15 parameters: 5 domains × 3 params)
        
    Returns:
        Predicted loss values [n_samples, 5] - one loss per domain per sample
    """
    
    # Ensure proportions is 2D: [n_samples, 5]
    proportions = np.atleast_2d(proportions)
    n_samples, n_domains = proportions.shape
    
    # Reshape params into [5 domains, 3 params per domain]
    if len(params) < 15:
        # Pad with ones if fewer than 15 parameters
        padded_params = np.ones(15)
        padded_params[:len(params)] = params
        params = padded_params
    else:
        params = params[:15]  # Truncate if more than 15
    
    params = params.reshape(5, 3)
    
    # Initialize output: [n_samples, 5]
    loss_predictions = np.zeros((n_samples, 5))
    
    # For each domain, compute loss using: Li = A_i + B_i * (∑_j proportion_ij^C_i)
    for domain_idx in range(5):
        A_i, B_i, C_i = params[domain_idx]  # 3 parameters for this domain
        
        # Add small epsilon to avoid zero and ensure positive values for power operation
        proportions_safe = proportions + 1e-8
        
        # Calculate ∑_j proportion_ij^C_i (sum of all proportions raised to power C_i)
        sum_powered_proportions = np.sum(np.power(proportions_safe, C_i), axis=1)
        
        # Apply the scaling law: Li = A_i + B_i * (∑_j proportion_ij^C_i)
        loss_predictions[:, domain_idx] = A_i + B_i * sum_powered_proportions
    
    return loss_predictions

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law to domain proportions and loss values
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters for 5 domains: A_i, B_i, C_i for each domain)
    """
    # Initialize 15 parameters (5 domains × 3 params per domain)
    # Initialize A_i to mean loss, B_i to 1.0, C_i to 1.0
    initial_params = np.ones(15)
    for i in range(5):
        initial_params[i * 3] = np.mean(loss_values[:, i])  # A_i = mean loss for domain i
        initial_params[i * 3 + 1] = 1.0  # B_i = 1.0
        initial_params[i * 3 + 2] = 1.0  # C_i = 1.0
    
    def objective(params):
        try:
            predicted = scaling_law_func(proportions, params)
            mse = np.mean((predicted - loss_values) ** 2)
            return mse
        except:
            return 1e6  # Return large error if computation fails
    
    # Use multiple optimization attempts with different starting points
    best_params = initial_params
    best_loss = float('inf')
    
    for attempt in range(3):
        # Random initialization for each attempt
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


# Set the number of parameters this function expects
scaling_law_func.num_params = 15

# EVOLVE-BLOCK-END