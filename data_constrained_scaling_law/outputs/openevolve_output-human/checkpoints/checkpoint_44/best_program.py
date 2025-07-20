"""
Human-designed Data-Constrained Scaling Law
L(N,D,U) = E + A/((U_N + U_N*R_N*(1-exp(-(N/U_N-1)/R_N)))^α) 
           + B/((U + U*R_D*(1-exp(-(D/U-1)/R_D)))^β)
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize, differential_evolution

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Human-designed data-constrained scaling law function.
    
    L(N,D,U) = E + A/((U_N + U_N*R_N*(1-exp(-(N/U_N-1)/R_N)))^α) 
               + B/((U + U*R_D*(1-exp(-(D/U-1)/R_D)))^β)
    
    7 parameters: [E, A, α, B, β, R_N, R_D]
    We assume U_N = model_size, U = unique_tokens.
    """
    E, A, alpha, B, beta, R_N, R_D = params
    # ensure arrays and positivity
    N = np.asarray(model_size, dtype=float) + 1e-12
    D = np.asarray(tokens, dtype=float) + 1e-12
    U = np.asarray(unique_tokens, dtype=float) + 1e-12
    U_N = N
    R_N = max(abs(R_N), 1e-12)
    R_D = max(abs(R_D), 1e-12)
    # compute first term
    ratio_N = np.clip((N / U_N - 1) / R_N, -50, 50)
    expN = 1 - np.exp(-ratio_N)
    denomN = np.maximum(U_N + U_N * R_N * expN, 1e-12)
    termN = A / np.power(denomN, alpha)
    # compute second term
    ratio_D = np.clip((D / U - 1) / R_D, -50, 50)
    expD = 1 - np.exp(-ratio_D)
    denomD = np.maximum(U + U * R_D * expD, 1e-12)
    termD = B / np.power(denomD, beta)
    return E + termN + termD

# EVOLVE-BLOCK-START
def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the data-constrained scaling law via a two-stage global-local multi-start search
    using log-transformed positive parameters for robust optimization.
    """
    # prepare data arrays
    D = np.asarray(tokens, dtype=float) + 1e-12
    N = np.asarray(model_size, dtype=float) + 1e-12
    U = np.asarray(unique_tokens, dtype=float) + 1e-12
    y = np.asarray(loss_values, dtype=float)
    
    # precompute data stats
    y_min, y_max = y.min(), y.max()
    span = max(y_max - y_min, 1e-8)
    
    # Transformation: x = [E, log(A), α, log(B), β, log(R_N), log(R_D)]
    def to_real(x):
        return np.array([
            x[0],
            np.exp(x[1]),
            x[2],
            np.exp(x[3]),
            x[4],
            np.exp(x[5]),
            np.exp(x[6])
        ], dtype=float)
    
    def obj(x):
        p = to_real(x)
        pred = scaling_law_func(D, N, U, p)
        return np.mean((pred - y) ** 2)
    
    # initial guess in x-space
    E0 = y_min
    x0 = np.array([
        E0,                    # E
        np.log(span),          # log(A)
        1.0,                   # α
        np.log(span * 0.5),    # log(B)
        1.0,                   # β
        np.log(1.0),           # log(R_N)
        np.log(1.0)            # log(R_D)
    ], dtype=float)
    
    # bounds in x-space
    lb_x = np.array([
        y_min * 0.5,               # E
        np.log(span * 1e-6),       # log(A)
        1e-3,                      # α
        np.log(span * 1e-6),       # log(B)
        1e-3,                      # β
        np.log(1e-6),              # log(R_N)
        np.log(1e-6)               # log(R_D)
    ], dtype=float)
    ub_x = np.array([
        y_max * 1.5,               # E
        np.log(span * 1e6),        # log(A)
        5.0,                       # α
        np.log(span * 1e6),        # log(B)
        5.0,                       # β
        np.log(100.0),             # log(R_N)
        np.log(100.0)              # log(R_D)
    ], dtype=float)
    bounds_x = list(zip(lb_x, ub_x))
    
    # Stage 1: global search (Differential Evolution)
    try:
        de_res = differential_evolution(
            obj, bounds_x,
            strategy='best1bin',
            maxiter=200,
            popsize=25,
            tol=1e-7,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True,
            seed=42,
            disp=False
        )
        x_global = np.clip(de_res.x, lb_x, ub_x)
    except Exception:
        x_global = x0.copy()
    
    # Stage 2: local multi-start refinement (L-BFGS-B) with several inits
    inits = [x_global, x0]
    rng = np.random.default_rng(123)
    for _ in range(4):
        rnd = lb_x + rng.random(len(lb_x)) * (ub_x - lb_x)
        inits.append(rnd)
    
    best_x = None
    best_loss = np.inf
    # local optimization settings
    local_opts = {'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
    # early stopping threshold
    threshold = (span * 0.005) ** 2
    
    for x_start in inits:
        try:
            res = minimize(
                obj, x_start,
                method='L-BFGS-B',
                bounds=bounds_x,
                options=local_opts
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                if best_loss < threshold:
                    break
        except Exception:
            continue
    
    # fallback if optimization failed
    if best_x is None:
        best_x = x_global.copy()
    
    # map back to real parameter space and clip to original plausible bounds
    p_best = to_real(best_x)
    # original bounds for clipping
    p_lb = np.array([lb_x[0], np.exp(lb_x[1]), lb_x[2], np.exp(lb_x[3]), lb_x[4], np.exp(lb_x[5]), np.exp(lb_x[6])])
    p_ub = np.array([ub_x[0], np.exp(ub_x[1]), ub_x[2], np.exp(ub_x[3]), ub_x[4], np.exp(ub_x[5]), np.exp(ub_x[6])])
    return np.clip(p_best, p_lb, p_ub)
# EVOLVE-BLOCK-END

# set expected parameter count
scaling_law_func.num_params = 7