# EVOLVE-BLOCK-START
"""
Domain mixture scaling law discovery for LLM training scenarios
Revised scaling law:
    L_i(r) = c_i + k_i * exp(−t_i * r_i)
– each domain’s loss depends exponentially on its own mixture proportion.
– Vectorized implementation for speed and clarity.
– 15 total parameters: for each of 5 domains, (c_i, k_i, t_i).
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    A vectorized exponential-decay scaling law.
    
    Li = c_i + k_i * exp(−t_i * r_i)
    
    Args:
        proportions: array [n_samples, 5], each row sums to 1.0
        params:      length-15 array, [c1,k1,t1, c2,k2,t2, …, c5,k5,t5]
    
    Returns:
        loss_predictions: [n_samples, 5]
    """
    # ensure array form
    P = np.atleast_2d(proportions).astype(float)
    n_samples, n_domains = P.shape
    
    # normalize or pad/truncate params to length 15
    p = np.array(params, dtype=float)
    if p.size < 15:
        tmp = np.ones(15, dtype=float)
        tmp[:p.size] = p
        p = tmp
    else:
        p = p[:15]
    # reshape to (5 domains × 3 params)
    p = p.reshape(5, 3)
    
    # unpack per-domain parameters
    c = p[:, 0][None, :]    # bias terms, shape (1,5)
    k = p[:, 1][None, :]    # scale terms, shape (1,5)
    t = p[:, 2][None, :]    # decay rates, shape (1,5)
    
    # compute loss: c_i + k_i * exp(−t_i * r_i)
    # using each domain's own proportion column
    L = c + k * np.exp(-t * P)
    return L

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law to domain proportions and loss values
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters for 5 domains: c_i, k_i, t_i)
    """
    # Initialize 15 parameters (5 domains × 3 params per domain)
    # Initialize c_i to mean loss, k_i to 1.0, t_i to 1.0
    initial_params = np.ones(15)
    for i in range(5):
        initial_params[i * 3] = np.mean(loss_values[:, i])  # c_i
        initial_params[i * 3 + 1] = 1.0  # k_i
        initial_params[i * 3 + 2] = 1.0  # t_i
    
    def objective(params):
        try:
            pred = scaling_law_func(proportions, params)
            return np.mean((pred - loss_values) ** 2)
        except:
            return 1e6
    
    # BFGS with a few random restarts
    best_params = initial_params.copy()
    best_loss = float('inf')
    for attempt in range(3):
        init = initial_params if attempt == 0 else initial_params + np.random.randn(15) * 0.1
        try:
            res = minimize(objective, init, method='BFGS')
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except:
            pass
    
    return best_params

# declare how many params the model uses
scaling_law_func.num_params = 15

# EVOLVE-BLOCK-END