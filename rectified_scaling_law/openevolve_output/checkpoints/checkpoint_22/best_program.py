# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Evolved to use a 4-parameter power‐law+offset form with robust
hybrid global-local optimization and NMSE objective.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    Four-parameter scaling law:
        loss(x) = a * (x + c)^(-b) + d
        
    where:
        a > 0      -- scale of the power law
        b > 0      -- exponent (steepness)
        c >= 0     -- horizontal shift (stabilizes small x)
        d >= 0     -- asymptotic floor (minimum loss)
    
    Args:
        data_points: array-like of training data sizes
        params:     array-like of exactly 4 parameters [a, b, c, d]
    
    Returns:
        Predicted loss values, same shape as data_points
    """
    a, b, c, d = params
    x = np.asarray(data_points, dtype=np.float64)
    # enforce numerical stability
    x = np.maximum(x, 1.0)
    return a * np.power(x + c, -b) + d

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law by minimizing normalized MSE (NMSE)
    over the data (hybrid global+local optimization).
    
    Args:
        data_points: array-like of training data sizes
        loss_values: array-like of observed losses
    
    Returns:
        params_opt: optimized parameters [a, b, c, d]
    """
    # Prepare arrays
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # Compute denominator for NMSE (variance of y)
    y_mean = np.mean(y)
    denom = np.mean((y - y_mean) ** 2) + 1e-12
    
    # Objective: normalized mean squared error
    def nmse_obj(params):
        preds = scaling_law_func(x, params)
        mse = np.mean((preds - y) ** 2)
        return mse / denom

    # Parameter bounds
    bounds = [
        (1e-8, np.max(y) * 10.0),  # a > 0
        (1e-8, 5.0),               # 0 < b <= 5
        (0.0, np.max(x) * 10.0),   # c >= 0
        (0.0, np.max(y))           # d >= 0
    ]

    # Smart initialization via log‐log linear fit
    with np.errstate(divide='ignore', invalid='ignore'):
        logx = np.log(x)
        logy = np.log(y)
    mask = np.isfinite(logx) & np.isfinite(logy)
    if np.sum(mask) > 2:
        # logy ≈ A + B*logx  =>  y ≈ exp(A) * x^B
        B, A = np.polyfit(logx[mask], logy[mask], 1)
        init_a = max(1e-6, np.exp(A))
        init_b = max(1e-6, -B)
    else:
        init_a, init_b = 1.0, 0.5
    init_c = max(0.0, np.min(x) * 0.1)
    init_d = max(0.0, np.min(y) * 0.1)
    initial_guess = [init_a, init_b, init_c, init_d]

    # Global search (Differential Evolution)
    try:
        de_result = differential_evolution(
            nmse_obj,
            bounds,
            strategy='best1bin',
            maxiter=500,
            popsize=12,
            tol=1e-6,
            polish=False,
            disp=False
        )
        global_best = de_result.x
    except Exception:
        global_best = initial_guess

    # Local refinement (L-BFGS-B)
    try:
        local_result = minimize(
            nmse_obj,
            x0=global_best,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol':1e-9}
        )
        params_opt = local_result.x if local_result.success else global_best
    except Exception:
        params_opt = global_best

    # Ensure exactly 4 params
    params_opt = np.asarray(params_opt, dtype=np.float64).flatten()
    if params_opt.shape[0] != 4:
        # pad or truncate (should not normally happen)
        p = np.ones(4, dtype=np.float64)
        p[:params_opt.shape[0]] = params_opt[:4]
        params_opt = p

    return params_opt

# Indicate to external code how many parameters we use
scaling_law_func.num_params = 4

# EVOLVE-BLOCK-END