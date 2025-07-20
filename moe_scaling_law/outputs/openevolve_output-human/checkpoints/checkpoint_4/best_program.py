"""
Human-designed MoE Scaling Law
log L(N,E) = a*log N + b*log Ê + c*log N * log Ê + d
where 1/Ê = 1/(E-1+(1/E_start - 1/E_max)) + 1/E_max
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize  # kept for backward compatibility
# Note: we'll import least_squares inside the fit function

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
    E_max = max(E_max, E_start + 1.0)  # Ensure E_max > E_start
    
    denominator = E - 1.0 + (1.0/E_start - 1.0/E_max)
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
    Fit the human-designed MoE scaling law to data using nonlinear least-squares
    with a log-domain residual and parameter transformations to enforce constraints.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters [a, b, c, d, E_start, E_max]
    """
    import numpy as np
    from scipy.optimize import least_squares
    
    # Convert inputs to arrays and safeguard
    E_arr = np.asarray(num_experts, dtype=float)
    N_arr = np.asarray(total_parameter_count, dtype=float)
    L_arr = np.asarray(loss_values, dtype=float)
    eps = 1e-8

    # Initial guesses for transformation parameters
    # E_start = 2.0 => u0 = log(E_start - 1)
    # E_max = 100.0 => v0 = log(E_max - E_start)
    u0 = np.log(2.0 - 1.0 + eps)
    v0 = np.log(100.0 - 2.0 + eps)

    # Compute initial E_start and E_max
    E_start0 = 1.0 + np.exp(u0)
    E_max0 = E_start0 + np.exp(v0)

    # Build initial features to regress a,b,c,d
    E_hat0 = scaling_law_func(E_arr, N_arr, [ -0.5, -0.5, 0.1, 0.0, E_start0, E_max0 ])
    logN = np.log(N_arr + eps)
    logE = np.log(np.maximum(E_hat0, eps))
    X = np.vstack([logN, logE, logN * logE, np.ones_like(logN)]).T
    y = np.log(L_arr + eps)
    # Linear least squares for a0,b0,c0,d0
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, b0, c0, d0 = sol

    # Pack initial x
    x0 = np.array([a0, b0, c0, d0, u0, v0], dtype=float)

    # Residual function in log-domain
    def residuals(x):
        a, b, c, d, u, v = x
        E_s = 1.0 + np.exp(u)
        E_m = E_s + np.exp(v)
        pred = scaling_law_func(E_arr, N_arr, [a, b, c, d, E_s, E_m])
        # Return log-pred - log-obs
        return np.log(pred + eps) - np.log(L_arr + eps)

    # Solve with robust least-squares
    result = least_squares(
        residuals,
        x0,
        loss='soft_l1',
        f_scale=0.1,
        max_nfev=5000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    # Extract optimized parameters
    a_opt, b_opt, c_opt, d_opt, u_opt, v_opt = result.x
    E_start_opt = 1.0 + np.exp(u_opt)
    E_max_opt = E_start_opt + np.exp(v_opt)

    return np.array([a_opt, b_opt, c_opt, d_opt, E_start_opt, E_max_opt], dtype=float)
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 6