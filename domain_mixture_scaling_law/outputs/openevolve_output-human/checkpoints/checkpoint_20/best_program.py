"""
Human-designed Domain Mixture Scaling Law
L_i(r) = c_i + k_i * exp(∑_{j=1}^M t_{ij} * r_j)
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares

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
    proportions = np.atleast_2d(proportions)
    n_samples, n_domains = proportions.shape
    
    # Ensure exactly 15 params
    if len(params) < 15:
        p = np.ones(15)
        p[:len(params)] = params
        params = p
    else:
        params = params[:15]
    
    params = params.reshape(5, 3)
    loss_predictions = np.zeros((n_samples, 5))
    
    for domain_idx in range(5):
        c_i, k_i, t_i = params[domain_idx]
        # ∑_j t_{ij} * r_j simplified as t_i * sum(r_j)
        sum_w = t_i * np.sum(proportions, axis=1)
        loss_predictions[:, domain_idx] = c_i + k_i * np.exp(sum_w)
    
    return loss_predictions

# EVOLVE-BLOCK-START
def fit_scaling_law(proportions, loss_values):
    """
    Fit the human-designed domain mixture scaling law to data using
    a bounded least-squares optimizer with multi-start restarts.
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters)
    """
    proportions = np.atleast_2d(proportions)
    loss_values = np.atleast_2d(loss_values)
    n_samples, n_domains = loss_values.shape

    # Initial guess: c_i = mean(loss_i), k_i = max-min (or small), t_i = 0
    means = np.mean(loss_values, axis=0)
    mins = np.min(loss_values, axis=0)
    maxs = np.max(loss_values, axis=0)
    init = np.zeros(15)
    for i in range(n_domains):
        init[i*3]     = means[i]
        init[i*3 + 1] = maxs[i] - mins[i] if maxs[i] > mins[i] else 0.1
        init[i*3 + 2] = 0.0

    # Define parameter bounds
    max_loss = maxs.max()
    c_lower, c_upper = 0.0, max_loss * 2.0 + 1.0
    k_lower, k_upper = 1e-8, max_loss * 10.0 + 1.0
    t_lower, t_upper = -5.0, 5.0

    lb = []
    ub = []
    for idx in range(15):
        mod = idx % 3
        if mod == 0:  # c_i
            lb.append(c_lower)
            ub.append(c_upper)
        elif mod == 1:  # k_i
            lb.append(k_lower)
            ub.append(k_upper)
        else:  # t_i
            lb.append(t_lower)
            ub.append(t_upper)
    lb = np.array(lb)
    ub = np.array(ub)

    # Residual function for least_squares
    def _residuals(params):
        pred = scaling_law_func(proportions, params)
        return (pred - loss_values).ravel()

    # Multi-start strategy
    best_params = init.copy()
    best_cost = np.inf
    # One deterministic start plus several random perturbations
    inits = [init] + [init + np.random.randn(15) * 0.1 for _ in range(4)]

    for x0 in inits:
        res = least_squares(
            _residuals,
            x0,
            bounds=(lb, ub),
            method='trf',
            jac='2-point',
            max_nfev=2000,
            xtol=1e-8,
            ftol=1e-8
        )
        if res.success and res.cost < best_cost:
            best_cost = res.cost
            best_params = res.x

    return best_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 15