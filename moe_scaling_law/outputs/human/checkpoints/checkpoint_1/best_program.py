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
    """
    a, b, c, d, E_start, E_max = params
    
    # Convert to numpy arrays and ensure positive values
    E = np.asarray(num_experts, dtype=float) + 1e-8
    N = np.asarray(total_parameter_count, dtype=float) + 1e-8
    
    # Enforce parameter ordering for stability
    E_start = max(E_start, 1.1)
    E_max = max(E_max, E_start + 1.0)
    
    # Compute Ê (E_hat)
    denom = E - 1.0 + 1.0 / (1.0 / E_start - 1.0 / E_max)
    denom = np.maximum(denom, 1e-8)
    one_over_E_hat = 1.0 / denom + 1.0 / E_max
    E_hat = 1.0 / np.maximum(one_over_E_hat, 1e-8)
    
    log_N = np.log(N)
    log_E_hat = np.log(np.maximum(E_hat, 1e-8))
    
    log_loss = a * log_N + b * log_E_hat + c * log_N * log_E_hat + d
    return np.exp(log_loss)

# EVOLVE-BLOCK-START
def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the human-designed MoE scaling law to data using a
    more robust initialization and L-BFGS-B with bounds.
    """
    E = np.asarray(num_experts, dtype=float) + 1e-8
    N = np.asarray(total_parameter_count, dtype=float) + 1e-8
    L = np.asarray(loss_values, dtype=float) + 1e-8

    # Preliminary linear regression on logs to estimate a, b, c, d
    # Use E_hat ≈ E for init
    log_N = np.log(N)
    log_E = np.log(E)
    X_lin = np.column_stack([log_N, log_E, log_N * log_E, np.ones_like(log_N)])
    y_lin = np.log(L)
    # Solve for [a, b, c, d] by least squares
    try:
        coeffs, *_ = np.linalg.lstsq(X_lin, y_lin, rcond=None)
        a0, b0, c0, d0 = coeffs
    except:
        a0, b0, c0, d0 = -0.5, -0.5, 0.1, 0.0

    # Initialize expert scaling params
    E_start0 = max(1.1, np.min(E))
    E_max0 = max(E_start0 + 1.0, np.max(E))

    init_params = np.array([a0, b0, c0, d0, E_start0, E_max0])

    # Bounds: a,b,c in [-10,10], d in [-20,20], E_start in [1.01, max(E)], E_max in [E_start+1, 1e6]
    # We enforce E_start>=1.01, E_max>=2.01
    bounds = [
        (-10.0, 10.0),  # a
        (-10.0, 10.0),  # b
        (-10.0, 10.0),  # c
        (-20.0, 20.0),  # d
        (1.01, np.max(E)),        # E_start
        (2.01, np.max(E) * 10.0)  # E_max
    ]

    # Objective: mean squared error on log-loss for stability
    def objective(params):
        # Enforce ordering inside objective to keep gradient stable
        ps = params.copy()
        ps[4] = max(ps[4], 1.01)
        ps[5] = max(ps[5], ps[4] + 1.0)
        pred = scaling_law_func(E, N, ps)
        # Compute error in log-space
        err = np.log(pred + 1e-12) - np.log(L)
        return np.mean(err * err)

    # Optimize with L-BFGS-B
    res = minimize(
        objective,
        init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if res.success:
        p_opt = res.x
        # Final enforce ordering
        p_opt[4] = max(p_opt[4], 1.01)
        p_opt[5] = max(p_opt[5], p_opt[4] + 1.0)
        return p_opt
    else:
        return init_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 6