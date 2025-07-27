# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios.

We retain a 4-parameter form:
    L(N) = a * (N + N0)^(-b) + c

and fit it by minimizing normalized MSE (NMSE) via a hybrid
global+local search: first Differential Evolution for global
exploration, then L-BFGS-B for fine-tuning. We enforce bounds
and add numerical safeguards for stability.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law:
        L(N) = a * (N + N0)^(-b) + c
    Args:
        data_points: array-like of training data sizes (N)
        params: array-like of 4 parameters [a, b, c, N0]
    Returns:
        numpy array of predicted losses
    """
    a, b, c, N0 = params
    x = np.asarray(data_points, dtype=np.float64)
    # enforce positivity and avoid zero-power issues
    x_eff = x + np.clip(N0, 0.0, None) + 1e-12
    return a * np.power(x_eff, -b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to observed (N, loss) pairs
    by minimizing NMSE via a hybrid global (DE) + local (L-BFGS-B) search.
    Args:
        data_points: array-like of training data sizes (N)
        loss_values: array-like of observed losses
    Returns:
        best-fit parameters [a, b, c, N0]
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # basic statistics
    N_max = max(np.max(x), 1.0)
    y_mean = np.mean(y)
    y_min, y_max = np.min(y), np.max(y)

    # initial c offset (asymptote) guess
    c0 = np.clip(y_min * 0.9, 1e-8, y_max)

    # parameter bounds: (a > 0), (b > 0), (c >= 0), (N0 >= 0)
    bounds = [
        (1e-8, y_max * 10.0),  # a
        (1e-8, 10.0),          # b
        (0.0, y_max),          # c
        (0.0, N_max * 2.0)     # N0
    ]

    # objective: normalized mean squared error
    def nmse_obj(p):
        pred = scaling_law_func(x, p)
        # if any invalid, penalize
        if not np.all(np.isfinite(pred)):
            return 1e6
        num = np.sum((pred - y) ** 2)
        den = np.sum((y - y_mean) ** 2) + 1e-12
        return num / den

    # 1) Heuristic initialization via log-log linear fit for a, b
    def heuristic_init(N0_guess):
        x_eff = x + N0_guess + 1e-12
        y_adj = np.clip(y - c0, 1e-8, None)
        logx = np.log(x_eff)
        logy = np.log(y_adj)
        slope, intercept = np.polyfit(logx, logy, 1)
        b_init = -slope
        a_init = np.exp(intercept)
        return np.array([a_init, b_init, c0, N0_guess], dtype=np.float64)

    # collect several initial guesses
    inits = []
    for f in (0.0, 0.01, 0.1, 0.5):
        inits.append(heuristic_init(N_max * f))

    # 2) Differential Evolution global search
    de_res = differential_evolution(
        nmse_obj, bounds,
        strategy='best1bin',
        maxiter=400, popsize=15,
        tol=1e-7, polish=False, seed=1234
    )
    best_p = de_res.x
    best_score = de_res.fun

    # compare with heuristic starts
    for init in inits:
        val = nmse_obj(init)
        if val < best_score:
            best_score, best_p = val, init

    # 3) Local refinement from best global/heuristic
    try:
        loc = minimize(
            nmse_obj, best_p,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        if loc.success and loc.fun < best_score:
            best_p, best_score = loc.x, loc.fun
    except Exception:
        pass

    # 4) Additional multi-start local from each heuristic
    for init in inits:
        try:
            loc = minimize(
                nmse_obj, init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if loc.success and loc.fun < best_score:
                best_p, best_score = loc.x, loc.fun
        except Exception:
            continue

    # fallback to safe defaults if something went wrong
    if not np.all(np.isfinite(best_p)):
        best_p = np.array([y_max, 0.5, c0, N_max * 0.01], dtype=np.float64)

    return best_p

# metadata: number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END