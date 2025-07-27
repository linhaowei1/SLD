import numpy as np
from scipy.optimize import minimize, differential_evolution

def scaling_law_func(data_points, params):
    """
    Predict loss according to a 4-parameter scaling law:
        L(N) = a * (N_norm + c)^(-b) + d
    where N_norm = N / N_max to improve numerical stability.

    Args:
        data_points: array-like, training data sizes
        params: array-like [a, b, c, d]

    Returns:
        numpy array of predicted losses
    """
    a, b, c, d = params
    x = np.asarray(data_points, dtype=np.float64)
    # Normalize inputs
    Nmax = x.max() if x.size else 1.0
    x_norm = x / (Nmax + 1e-12)
    # Enforce non-negativity
    c_pos = max(c, 0.0)
    xp = x_norm + c_pos
    xp = np.maximum(xp, 1e-12)
    return a * np.power(xp, -b) + d

# expose number of parameters
scaling_law_func.num_params = 4


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law L(N) = a*(N_norm + c)^(-b) + d
    by minimizing the normalized mean squared error (NMSE) via
    a hybrid global-local strategy.
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)

    # Stats for normalization and bounds
    Nmax = x.max() if x.size else 1.0
    y_min, y_max = float(y.min()), float(y.max())
    y_range = max(y_max - y_min, 1e-8)
    denom = np.sum((y - y.mean())**2) + 1e-12

    # Objective: normalized MSE
    def obj_fn(params):
        # simple bounds check to reject invalid regions
        a, b, c, d = params
        if a <= 0 or b <= 0 or c < 0 or d < 0 or d > y_min:
            return np.inf
        pred = scaling_law_func(x, params)
        if not np.all(np.isfinite(pred)):
            return np.inf
        return np.sum((pred - y)**2) / denom

    # Parameter bounds: a>0, b>0, c>=0, d in [0, y_min]
    bounds = [
        (1e-8, 10.0 * y_range),  # a
        (1e-6, 10.0),            # b
        (0.0, 1.0),              # c (in normalized units)
        (0.0, y_min)             # d
    ]

    # Heuristic initialization via log-log linear fit for y - y_min
    with np.errstate(divide='ignore', invalid='ignore'):
        x_norm = x / (Nmax + 1e-12)
        log_x = np.log(x_norm + 1e-8)
        log_y = np.log(np.maximum(y - y_min + 1e-8, 1e-8))
        slope, intercept = np.polyfit(log_x, log_y, 1)
    b0 = max(-slope, 1e-3)
    a0 = max(np.exp(intercept), 1e-8)
    d0 = y_min

    # Build initial guess list: a few c0 variants + some random
    init_list = [
        [a0, b0, 0.0, d0],
        [a0, b0, 0.05, d0],
        [a0, b0, 0.1, d0]
    ]
    rng = np.random.RandomState(42)
    for _ in range(5):
        init_list.append([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1]),
            rng.uniform(bounds[3][0], bounds[3][1]),
        ])

    best_score = np.inf
    best_params = None

    # 1) Global search: Differential Evolution
    try:
        de_res = differential_evolution(
            obj_fn, bounds,
            strategy='best1bin',
            maxiter=50, popsize=15,
            tol=1e-6, polish=True, disp=False
        )
        if de_res.fun < best_score:
            best_score, best_params = de_res.fun, de_res.x.copy()
    except Exception:
        pass

    # 2) Local refinement: L-BFGS-B from multiple starts
    for init in init_list:
        try:
            res = minimize(
                obj_fn, x0=init, method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 500}
            )
            if res.success and res.fun < best_score:
                best_score, best_params = res.fun, res.x.copy()
                # early exit if extremely good fit
                if best_score < 1e-10:
                    break
        except Exception:
            continue

    # 3) Fallback: simple power-law fit if all else fails
    if best_params is None:
        # Fit log(y) = log(a) - b*log(x_norm + eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_x = np.log(x_norm + 1e-8)
            log_y = np.log(y + 1e-8)
            slope, intercept = np.polyfit(log_x, log_y, 1)
        b_f = max(-slope, 1e-3)
        a_f = max(np.exp(intercept), 1e-8)
        best_params = np.array([a_f, b_f, 0.0, 0.0], dtype=np.float64)

    return np.array(best_params, dtype=np.float64)