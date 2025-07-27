import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given mixture proportions and params.
    proportions: array [n_samples,5], rows sum to 1.
    params: flat array length 35 (5 domains × 7 params).
            For domain i: [a_i, b_i, c_i, d_i, e_i, f_i, g_i].
    Returns: array [n_samples,5] of predicted losses.
    """
    p = np.asarray(proportions, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    n, d = p.shape
    assert d == 5, "Proportions must have shape [n,5]"
    eps = 1e-12

    # Reshape parameters into (5 domains × 7 params)
    P = np.asarray(params, dtype=float).reshape(5, 7)

    # Precompute mixture entropy H and second‐moment S = ∑ p_j^2
    H = -np.sum(p * np.log(p + eps), axis=1, keepdims=True)  # (n,1)
    S = np.sum(p * p, axis=1, keepdims=True)                # (n,1)

    # Unpack params
    # a_i: baseline, b_i & c_i: self‐power, d_i & e_i: complement‐power,
    # f_i: entropy weight, g_i: cross‐moment weight
    A = P[:, 0][None, :]  # (1,5)
    B = P[:, 1][None, :]
    C = P[:, 2][None, :]
    D = P[:, 3][None, :]
    E = P[:, 4][None, :]
    F = P[:, 5][None, :]
    G = P[:, 6][None, :]

    # Self‐power term: b_i * p_i^c_i
    term_self = B * np.power(p + eps, C)
    # Complement‐power term: d_i * (1 - p_i)^e_i
    term_comp = D * np.power(1.0 - p + eps, E)
    # Entropy coupling: f_i * H
    term_ent = F * H
    # Cross‐domain squared interactions: g_i * (S - p_i^2)
    term_cross = G * (S - p * p)

    # Sum all terms
    L_pred = A + term_self + term_comp + term_ent + term_cross
    return L_pred

def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35 parameters of the scaling law by minimizing
    MSE in log‐loss space + L2 regularization.
    """
    p = np.asarray(proportions, dtype=float)
    L_obs = np.asarray(loss_values, dtype=float)
    n, d = p.shape
    assert d == 5 and L_obs.shape == (n, 5), "Invalid input shapes"
    eps = 1e-8

    # Initialize parameters
    L_mean = np.maximum(L_obs.mean(axis=0), eps)
    init = np.zeros(35, dtype=float)
    for i in range(5):
        base = L_mean[i]
        idx = 7 * i
        init[idx + 0] = base * 0.5  # a_i
        init[idx + 1] = base        # b_i
        init[idx + 2] = 1.0         # c_i
        init[idx + 3] = base * 0.5  # d_i
        init[idx + 4] = 1.0         # e_i
        init[idx + 5] = 0.0         # f_i
        init[idx + 6] = 0.0         # g_i

    # Bounds: force exponents positive for stability
    bounds = []
    for _ in range(5):
        bounds += [
            (None, None),    # a_i
            (1e-6, None),    # b_i >= 0
            (1e-3, 10.0),    # c_i > 0
            (None, None),    # d_i
            (1e-3, 10.0),    # e_i > 0
            (None, None),    # f_i
            (None, None),    # g_i
        ]

    def objective(params):
        L_pred = scaling_law_func(p, params)
        # log‐MSE to reduce heteroscedasticity
        diff = np.log(L_pred + eps) - np.log(L_obs + eps)
        mse_log = np.mean(diff * diff)
        reg = 1e-6 * np.sum(params * params)
        return mse_log + reg

    best_params = init.copy()
    best_val = objective(best_params)

    # Multi‐start optimization
    for seed in range(5):
        x0 = init + (np.random.randn(35) * 0.1 if seed > 0 else 0)
        try:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-9}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return best_params
# EVOLVE-BLOCK-END

# Attach metadata
scaling_law_func.num_params = 35