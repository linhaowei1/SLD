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
    Fit the human-designed MoE scaling law to data by minimizing log-loss residuals.
    
    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (6 parameters)
    """
    import numpy as _np
    from scipy.optimize import least_squares

    # Ensure arrays and small epsilon for stability
    E = _np.asarray(num_experts, dtype=float) + 1e-8
    N = _np.asarray(total_parameter_count, dtype=float) + 1e-8
    L = _np.asarray(loss_values, dtype=float) + 1e-8

    # Work in log-space to stabilize fitting
    y = _np.log(L)

    # Initial linear regression to estimate [a, b, c, d] ignoring expert-scaling nuances
    X = _np.column_stack([
        _np.log(N),
        _np.log(E),
        _np.log(N) * _np.log(E),
        _np.ones_like(N)
    ])
    try:
        coef, *_ = _np.linalg.lstsq(X, y, rcond=None)
        a0, b0, c0, d0 = coef
    except Exception:
        # Fallback defaults
        a0, b0, c0, d0 = -0.5, -0.5, 0.0, _np.mean(y)

    # Reasonable starting points for expert parameters
    E_start0 = max(2.0, float(_np.median(E)))
    E_max0 = max(E_start0 + 1.0, float(_np.max(E)))

    initial = _np.array([a0, b0, c0, d0, E_start0, E_max0])

    # Define residual in log-space
    def residual(params):
        pred = scaling_law_func(E, N, params)
        return _np.log(pred + 1e-8) - y

    # Bounds: [a,b,c,d] in [-10,10], E_start>1, E_max>2
    lower = _np.array([-10, -10, -10, -10, 1.01, 2.01])
    upper = _np.array([10, 10, 10, 10, _np.max(E)*10, _np.max(E)*20 + 10])

    try:
        res = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method='trf',
            loss='huber',
            f_scale=0.1,
            max_nfev=2000
        )
        return res.x if res.success else initial
    except Exception:
        return initial
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 6