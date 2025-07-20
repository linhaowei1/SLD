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
    Fit the human-designed domain mixture scaling law to data using constrained least-squares.
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters)
    """
    from scipy.optimize import least_squares
    
    proportions = np.atleast_2d(proportions)
    loss_values = np.atleast_2d(loss_values)
    n_samples, n_domains = loss_values.shape
    assert n_domains == 5, "Expecting 5 domains in loss_values"
    
    # Initialize parameters: c_i ~ 90% min loss, k_i ~ half range, t_i = 0
    init_params = []
    lb, ub = [], []
    for i in range(5):
        li = loss_values[:, i]
        c0 = max(np.min(li) * 0.9, 1e-6)
        k0 = max((np.max(li) - np.min(li)) * 0.5, 1e-6)
        t0 = 0.0
        init_params += [c0, k0, t0]
        # Bounds: c_i >= 0, <= 2*max; k_i >= 0, <= 10*range; t_i in [-5,5]
        lb += [0.0, 0.0, -5.0]
        ub += [np.max(li) * 2.0, (np.max(li) - np.min(li)) * 10.0 + 1e-6, 5.0]
    
    init_params = np.array(init_params)
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Define residuals
    def residuals(params):
        pred = scaling_law_func(proportions, params)
        return (pred - loss_values).ravel()
    
    # Solve using Trust Region Reflective algorithm for bounded least-squares
    result = least_squares(
        residuals,
        init_params,
        bounds=(lb, ub),
        method='trf',
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=2000
    )
    
    # If optimize fails, fallback to simple BFGS with original init
    if not result.success:
        def obj(params):
            p = scaling_law_func(proportions, params)
            return np.mean((p - loss_values)**2)
        bfgs = minimize(obj, init_params, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        return bfgs.x if bfgs.success else init_params
    
    return result.x
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 15