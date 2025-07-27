# EVOLVE-BLOCK-START
"""
Domain mixture scaling law discovery for LLM training scenarios
Improved form: each domain loss combines a bias term, a power‐law term on its own proportion,
and a mixture entropy term to capture cross‐domain interactions.
Li = a_i + b_i * p_i^(-c_i) + d_i * H
where H = -∑_j p_j log(p_j)

Per‐domain params: a_i, b_i, c_i, d_i (4 params × 5 domains = 20 parameters)
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Compute predicted losses for 5 domains given mixture proportions.

    Args:
        proportions: array [n_samples, 5], rows sum to 1
        params:      flat array of length 20, reshaped to [5,4] as [a,b,c,d] per domain

    Returns:
        losses: array [n_samples, 5]
    """
    proportions = np.atleast_2d(proportions)
    n, D = proportions.shape
    assert D == 5, "Expected 5 domain proportions"
    # Reshape params into (5 domains × 4 parameters)
    p = params.reshape(5, 4)  # [[a_i, b_i, c_i, d_i], ...]
    a, b, c, d = p[:,0], p[:,1], p[:,2], p[:,3]

    # Numerical safety
    eps = 1e-12
    P_safe = np.clip(proportions, eps, 1.0)  # [n,5]
    # Entropy term H per sample
    H = -np.sum(P_safe * np.log(P_safe), axis=1)  # [n]

    # Power‐law term on each domain's own proportion
    # shape: [n,5], c broadcast over rows
    power_term = P_safe ** (-c[np.newaxis, :])

    # Compose loss: broadcast a, b, d over samples
    losses = (
        a[np.newaxis, :] +
        b[np.newaxis, :] * power_term +
        d[np.newaxis, :] * H[:, np.newaxis]
    )
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law parameters to observed losses.

    Args:
        proportions: array [n_samples, 5]
        loss_values: array [n_samples, 5]

    Returns:
        flat array of optimized parameters (length 20)
    """
    P = np.atleast_2d(proportions)
    L = np.atleast_2d(loss_values)
    n, D = P.shape
    assert D == 5 and L.shape == (n, 5)

    # Initialize parameters:
    # a_i ~ mean observed loss for domain i
    # b_i, c_i ~ 1.0, d_i ~ std of losses (scale of entropy effect)
    a0 = np.mean(L, axis=0)
    b0 = np.ones(5)
    c0 = np.ones(5)
    d0 = np.ones(5) * np.std(L)
    init_params = np.stack([a0, b0, c0, d0], axis=1).ravel()  # length 20

    # Bounds: a_i free, b_i >= tiny, c_i >= tiny, d_i >= 0
    bounds = []
    for _ in range(5):
        bounds += [(None, None),    # a_i
                   (1e-8, None),    # b_i
                   (1e-8, None),    # c_i
                   (0.0, None)]     # d_i

    def objective(x):
        pred = scaling_law_func(P, x)
        return np.mean((pred - L) ** 2)

    result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds)

    return result.x if result.success else result.x

# Expose expected param count
scaling_law_func.num_params = 20
# EVOLVE-BLOCK-END