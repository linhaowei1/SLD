# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Optimized 4-parameter offset power-law with targeted 2D global search
on (B, x0), closed‐form estimation of (A, α), and multi-start L-BFGS-B refinement.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    Offset power-law form:
        L(N) = A * (N + x0)^(-alpha) + B
    params: [A, alpha, x0, B]
    """
    A, alpha, x0, B = params
    x = np.asarray(data_points, dtype=float)
    x_safe = np.maximum(x + x0, 1e-16)
    return A * np.power(x_safe, -alpha) + B

def fit_scaling_law(data_points, loss_values):
    """
    Fit the offset power-law scaling law:
      1) Global 2D search over B and x0 via differential evolution,
         with closed-form linear regression for A and alpha.
      2) Multi-start L-BFGS-B refinement on all 4 parameters.
    Objective: minimize normalized MSE (NMSE).
    """

    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    n = len(x)

    x_max = np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Bounds for B and x0 in DE
    B_up = max(y_min * 0.9, 1e-6)
    B_bounds = (0.0, B_up)
    x0_bounds = (0.0, x_max)

    # Objective for DE: given [B, x0], estimate A and alpha via log-linear fit
    def obj_Bx0(bx):
        B_val, x0_val = float(bx[0]), float(bx[1])
        y_adj = y - B_val
        if x0_val < 0 or np.any(y_adj <= 0):
            return 1e6
        xa = x + x0_val
        if np.any(xa <= 0):
            return 1e6
        log_x = np.log(xa)
        log_y = np.log(y_adj)
        # slope m, intercept c
        m, c = np.polyfit(log_x, log_y, 1)
        alpha = -m
        A = np.exp(c)
        if A <= 0 or alpha <= 0:
            return 1e6
        y_pred = A * np.power(xa, -alpha) + B_val
        return np.sum((y_pred - y) ** 2) / np.sum(y ** 2)

    # 1) 2D global search for B, x0
    try:
        de_res = differential_evolution(
            obj_Bx0,
            bounds=[B_bounds, x0_bounds],
            strategy='best1bin',
            maxiter=60,
            popsize=12,
            tol=1e-5,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=123,
            polish=True,
            disp=False
        )
        B0, x0_0 = de_res.x
    except Exception:
        B0, x0_0 = 0.0, 0.0

    # Closed-form estimates for A and alpha
    y_adj = y - B0
    mask = y_adj > 0
    if mask.sum() >= 2:
        xa = x + x0_0
        log_x = np.log(xa[mask])
        log_y = np.log(y_adj[mask])
        m, c = np.polyfit(log_x, log_y, 1)
        alpha0 = max(-m, 1e-6)
        A0 = max(np.exp(c), 1e-6)
    else:
        A0, alpha0 = y_max, 1.0

    init_params = np.array([A0, alpha0, x0_0, B0], dtype=float)

    # Bounds for all parameters: (A, alpha, x0, B)
    bounds = [
        (1e-12, y_max * 10.0),   # A
        (1e-6,    10.0),         # alpha
        (0.0,     x_max),        # x0
        (0.0,     y_max)         # B
    ]

    # NMSE objective for full 4-parameter fit
    def nmse(params):
        y_pred = scaling_law_func(x, params)
        return np.sum((y_pred - y) ** 2) / np.sum(y ** 2)

    # Multi-start local refinement
    best_params = init_params.copy()
    best_score = nmse(best_params)
    rng = np.random.default_rng(456)
    starts = [best_params]
    for _ in range(4):
        perturb = rng.normal(1.0, 0.1, size=4)
        trial = best_params * perturb
        # clip within bounds
        for i, (lo, hi) in enumerate(bounds):
            trial[i] = np.clip(trial[i], lo + 1e-12, hi - 1e-12)
        starts.append(trial)

    for start in starts:
        try:
            res = minimize(
                nmse,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 1000, 'disp': False}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            continue

    return best_params

# annotate number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END