"""
Human-designed MoE Scaling Law
log L(N,E) = a*log N + b*log Ê + c*log N * log Ê + d
where 1/Ê = 1/(E-1+(1/E_start - 1/E_max)) + 1/E_max
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize  # kept for compatibility, not used directly below

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
    
    # Enforce E_start > 1 and E_max > E_start
    E_start = max(E_start, 1.0001)
    E_max = max(E_max, E_start + 1e-3)
    
    # Compute Ê (E_hat)
    denom = E - 1.0 + (1.0 / E_start - 1.0 / E_max)
    denom = np.maximum(denom, 1e-8)
    one_over_E_hat = 1.0 / denom + 1.0 / E_max
    E_hat = 1.0 / np.maximum(one_over_E_hat, 1e-8)
    
    # Compute logs
    log_N = np.log(N)
    log_Ehat = np.log(np.maximum(E_hat, 1e-8))
    
    # Scaling law in log-domain
    log_loss = a * log_N + b * log_Ehat + c * log_N * log_Ehat + d
    return np.exp(log_loss)

# EVOLVE-BLOCK-START
def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the human-designed MoE scaling law to data using robust least-squares
    on the log-loss residuals, with enforced constraints E_start>1 and E_max>E_start.
    
    Args:
        num_experts: Array-like of number of experts
        total_parameter_count: Array-like of total parameter counts
        loss_values: Array-like of observed loss values
        
    Returns:
        Numpy array of optimized parameters [a, b, c, d, E_start, E_max]
    """
    import numpy as np
    from scipy.optimize import least_squares

    # Data preparation
    E = np.asarray(num_experts, dtype=float)
    N = np.asarray(total_parameter_count, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    y = np.clip(y, 1e-8, None)
    logy = np.log(y)
    
    # Initial linear regression in log-space to get [a,b,c,d]
    L1 = np.log(N)
    L2 = np.log(E + 1e-8)
    X = np.vstack([L1, L2, L1 * L2, np.ones_like(L1)]).T
    # Solve X * [a,b,c,d] = logy in least-squares sense
    coeffs, *_ = np.linalg.lstsq(X, logy, rcond=None)
    a0, b0, c0, d0 = coeffs

    # Heuristic initial E_start and E_max based on E distribution
    E_sorted = np.sort(E)
    E_start0 = max(1.1, np.quantile(E_sorted, 0.10))
    E_max0 = max(E_start0 + 1.0, np.quantile(E_sorted, 0.90))
    u0 = np.log(E_start0 - 1.0)         # transform for E_start = 1 + exp(u)
    v0 = np.log(E_max0 - E_start0)      # transform for E_max = E_start + exp(v)

    x0 = np.array([a0, b0, c0, d0, u0, v0], dtype=float)
    
    # Residuals: difference in log-space
    def residuals(x):
        a, b, c, d, u, v = x
        E_start = 1.0 + np.exp(u)
        E_max = E_start + np.exp(v)
        preds = scaling_law_func(E, N, [a, b, c, d, E_start, E_max])
        preds = np.clip(preds, 1e-8, None)
        return np.log(preds) - logy

    # Bounds: u >= log(1e-3), v >= log(1e-3) to ensure positivity
    lower = [-np.inf, -np.inf, -np.inf, -np.inf, np.log(1e-3), np.log(1e-3)]
    upper = [ np.inf] * 6

    # Robust least-squares with Huber loss
    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        loss='huber',
        f_scale=0.1,
        method='trf',
        max_nfev=5000,
        verbose=0
    )

    # Unpack optimized values
    a_opt, b_opt, c_opt, d_opt, u_opt, v_opt = result.x
    E_start_opt = 1.0 + np.exp(u_opt)
    E_max_opt = E_start_opt + np.exp(v_opt)

    return np.array([a_opt, b_opt, c_opt, d_opt, E_start_opt, E_max_opt], dtype=float)
# EVOLVE-BLOCK-END

# Declare expected number of parameters
scaling_law_func.num_params = 6