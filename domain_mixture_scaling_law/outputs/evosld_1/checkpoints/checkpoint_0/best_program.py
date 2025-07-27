# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses from mixture proportions using a 5-parameter power-law model per domain:
      Li = A_i + B_i * p_i^( -C_i ) + D_i * (1 - p_i)^( E_i )
    where B_i, C_i, D_i, E_i are constrained positive via exp-transforms.

    Args:
      proportions: array [n_samples, 5], each row sums to 1.0
      params:       flat array of length 25 (5 domains × 5 params). If shorter,
                    it is zero-padded; if longer it is truncated.

    Returns:
      preds: array [n_samples, 5] of predicted losses.
    """
    proportions = np.atleast_2d(proportions)
    n_samples, n_domains = proportions.shape
    assert n_domains == 5, "Expect 5 domains"

    # Ensure correct param length
    P = 5 * 5
    p = np.zeros(P, dtype=float)
    p[:min(len(params), P)] = params[:P]
    p = p.reshape(5, 5)

    eps = 1e-8
    preds = np.zeros((n_samples, 5), dtype=float)
    for i in range(5):
        A_i = p[i, 0]
        B_i = np.exp(p[i, 1])
        C_i = np.exp(p[i, 2])
        D_i = np.exp(p[i, 3])
        E_i = np.exp(p[i, 4])

        pi = proportions[:, i] + eps
        qi = 1.0 - proportions[:, i] + eps

        preds[:, i] = A_i + B_i * np.power(pi, -C_i) + D_i * np.power(qi, E_i)

    return preds

def fit_scaling_law(proportions, loss_values):
    """
    Fit the above scaling law via MSE minimization.
    Initializes A_i to the mean loss, logs of other params to zero.
    Uses L-BFGS-B with up to 3 random restarts.
    """
    proportions = np.atleast_2d(proportions)
    loss_values = np.atleast_2d(loss_values)
    n_samples, n_domains = proportions.shape
    assert n_domains == 5 and loss_values.shape == (n_samples, 5)

    # Initial parameters: [A_i, log B_i, log C_i, log D_i, log E_i] per domain
    init = np.zeros(5 * 5, dtype=float)
    for i in range(5):
        init[5 * i + 0] = np.mean(loss_values[:, i])  # A_i
        # logs of B_i, C_i, D_i, E_i remain 0 => initial B=C=D=E=1

    def objective(x):
        pred = scaling_law_func(proportions, x)
        return np.mean((pred - loss_values) ** 2)

    best_params = init.copy()
    best_loss = objective(init)

    for attempt in range(3):
        x0 = init if attempt == 0 else init + np.random.randn(init.size) * 0.1
        try:
            res = minimize(objective, x0, method='L-BFGS-B')
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params

# Expose expected parameter count
scaling_law_func.num_params = 25
# EVOLVE-BLOCK-END