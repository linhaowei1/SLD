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
    E = np.asarray(num_experts, dtype=float) + 1e-12
    N = np.asarray(total_parameter_count, dtype=float) + 1e-12

    # Enforce E_start > 1 and E_max > E_start
    E_start = max(E_start, 1.0001)
    E_max = max(E_max, E_start + 1e-6)

    # Compute Ê (E_hat)
    denom = E - 1.0 + (1.0/E_start - 1.0/E_max)
    denom = np.maximum(denom, 1e-12)
    inv_Ehat = 1.0/denom + 1.0/E_max
    E_hat = 1.0/np.maximum(inv_Ehat, 1e-12)

    # Compute log terms
    logN = np.log(N)
    logEhat = np.log(np.maximum(E_hat, 1e-12))

    # Scaling law in log-domain
    log_loss = a*logN + b*logEhat + c*logN*logEhat + d

    # Back to original loss
    return np.exp(log_loss)


# EVOLVE-BLOCK-START
def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the human-designed MoE scaling law to data using nonlinear least-squares
    with log-domain residuals and parameter transforms for constraints.

    Args:
        num_experts: Array of number of experts
        total_parameter_count: Array of total parameter counts
        loss_values: Array of corresponding loss values

    Returns:
        Optimized parameters [a, b, c, d, E_start, E_max]
    """
    import numpy as np
    from scipy.optimize import least_squares

    # Prepare data arrays
    E = np.asarray(num_experts, dtype=float)
    N = np.asarray(total_parameter_count, dtype=float)
    L = np.asarray(loss_values, dtype=float)
    eps = 1e-12

    # Initial E_start, E_max based on data range
    E_min = max(E.min(), 1.1)
    E_max_dat = E.max()
    E_start0 = E_min
    E_max0 = E_max_dat if E_max_dat > E_start0 + 0.1 else E_start0 + 0.1
    u0 = np.log(E_start0 - 1.0 + eps)
    v0 = np.log(E_max0 - E_start0 + eps)

    # Linear regression for initial [a, b, c, d]
    # Use a trial parameter set for Ê
    trial_params = [-1.0, -0.5, 0.1, 0.0, E_start0, E_max0]
    Ehat0 = scaling_law_func(E, N, trial_params)
    logN = np.log(N + eps)
    logE = np.log(np.maximum(Ehat0, eps))
    X = np.vstack([logN, logE, logN*logE, np.ones_like(logN)]).T
    y = np.log(L + eps)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, b0, c0, d0 = sol

    # Pack initial guess
    x0 = np.array([a0, b0, c0, d0, u0, v0], dtype=float)

    # Residuals in log-domain
    def residuals(x):
        a, b, c, d, u, v = x
        E_s = 1.0 + np.exp(u)
        E_m = E_s + np.exp(v)
        pred = scaling_law_func(E, N, [a, b, c, d, E_s, E_m])
        return np.log(pred + eps) - np.log(L + eps)

    # Bounds: ensure u, v stay in a numerically safe range
    lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf, np.log(eps), np.log(eps)])
    upper = np.array([ np.inf,  np.inf,  np.inf,  np.inf, np.log(1e6), np.log(1e6)])

    # Run robust least-squares
    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        loss='huber',
        f_scale=0.05,
        jac='2-point',
        max_nfev=20000,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14
    )

    # Decode parameters
    a_opt, b_opt, c_opt, d_opt, u_opt, v_opt = result.x
    E_start_opt = 1.0 + np.exp(u_opt)
    E_max_opt = E_start_opt + np.exp(v_opt)

    return np.array([a_opt, b_opt, c_opt, d_opt, E_start_opt, E_max_opt], dtype=float)
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 6