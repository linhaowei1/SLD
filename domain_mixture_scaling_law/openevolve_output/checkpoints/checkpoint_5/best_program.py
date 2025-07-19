# EVOLVE-BLOCK-START
"""
Domain mixture scaling law discovery for LLM training scenarios
Improved model with separate in-domain and out-of-domain contributions.

Scaling Law per domain i:
    L_i = A_i 
          + B_i * (p_i + ε)^{C_i}          # in-domain scaling
          + D_i * (1 - p_i + ε)^{E_i}      # out-of-domain scaling

Each domain uses 5 parameters: A_i, B_i, C_i, D_i, E_i (≤7 allowed).
Total parameters = 5 domains × 5 params = 25.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given mixture proportions.

    Args:
        proportions: array [n_samples, 5], rows sum to 1.0
        params:      flat array length 25 (5 domains × 5 params)

    Returns:
        losses: array [n_samples, 5]
    """
    p = np.atleast_2d(proportions)
    n_samples, n_domains = p.shape
    # Extract and reshape parameters
    params = np.array(params, dtype=float)[:n_domains * 5].reshape(n_domains, 5)
    A = params[:, 0]  # bias
    B = params[:, 1]  # in-domain scale
    C = params[:, 2]  # in-domain exponent
    D = params[:, 3]  # out-of-domain scale
    E = params[:, 4]  # out-of-domain exponent

    # Safe proportions
    eps = 1e-8
    p_safe = p + eps
    q_safe = 1.0 - p + eps  # out-of-domain mass

    # Compute term shapes: (n_samples, 5)
    term_in  = B * np.power(p_safe, C)
    term_out = D * np.power(q_safe, E)

    # Broadcast A: shape (5,) -> (n_samples, 5)
    losses = A + term_in + term_out
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law parameters by minimizing MSE.

    Args:
        proportions:  array [n_samples, 5]
        loss_values:  array [n_samples, 5]

    Returns:
        best_params: flat array length 25
    """
    p = np.atleast_2d(proportions)
    y = np.atleast_2d(loss_values)
    n_samples, n_domains = p.shape
    assert y.shape == (n_samples, n_domains), "Shape mismatch"

    num_params = n_domains * 5
    # Initialize: A_i = mean loss, others = 1.0
    init = np.zeros(num_params, dtype=float)
    for i in range(n_domains):
        init[i*5 + 0] = np.mean(y[:, i])  # A_i
        init[i*5 + 1] = 1.0               # B_i
        init[i*5 + 2] = 1.0               # C_i
        init[i*5 + 3] = 1.0               # D_i
        init[i*5 + 4] = 1.0               # E_i

    # Bounds to keep exponents positive and scales non-negative
    bounds = []
    for _ in range(n_domains):
        bounds += [
            (None,    None),   # A_i unbounded
            (0.0,     None),   # B_i ≥ 0
            (0.1,     10.0),   # C_i ∈ [0.1, 10]
            (0.0,     None),   # D_i ≥ 0
            (0.1,     10.0)    # E_i ∈ [0.1, 10]
        ]

    def mse_obj(params):
        pred = scaling_law_func(p, params)
        return np.mean((pred - y) ** 2)

    best_loss = np.inf
    best_params = init.copy()
    # Multiple restarts to avoid local minima
    for restart in range(5):
        if restart == 0:
            x0 = init.copy()
        else:
            x0 = init + np.random.randn(num_params) * 0.1
        res = minimize(mse_obj, x0, method='L-BFGS-B', bounds=bounds)
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    return best_params

# Expose expected parameter count
scaling_law_func.num_params = 25
# EVOLVE-BLOCK-END