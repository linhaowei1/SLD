import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
"""
Enhanced 7-parameter‐per‐domain Domain Mixture Scaling Law

For each domain i we model:
    L_i = A_i
          + B_i * (p_i + ε)^(−C_i)
          + D_i * (1 − p_i + ε)^(−E_i)
          + F_i * H
          + G_i * p_i * (1 − p_i)

where:
  * p_i is the domain‐i proportion
  * H = −∑_j p_j log(p_j + ε) is the mixture entropy
  * [A_i,B_i,C_i,D_i,E_i,F_i,G_i] are 7 parameters per domain
Total parameters = 5 domains × 7 = 35.
"""

def scaling_law_func(proportions, params):
    """
    Predict per‐domain losses from mixture proportions.

    Args:
      proportions: array-like, shape (n_samples, 5)
        Rows sum to 1.
      params: array-like, shape (35,)
        Domain parameters flattened: [A1..A5, B1..B5, C1..C5,
                                      D1..D5, E1..E5, F1..F5, G1..G5]

    Returns:
      losses: ndarray, shape (n_samples, 5)
    """
    p = np.atleast_2d(proportions).astype(float)
    n, d = p.shape
    assert d == 5, "Expect 5 domain proportions"
    eps = 1e-8

    # Safe proportions for log/power
    p_safe = np.clip(p, eps, 1.0)
    one_minus = np.clip(1.0 - p, eps, 1.0)
    # Mixture entropy H per sample
    H = -np.sum(p_safe * np.log(p_safe), axis=1, keepdims=True)  # (n,1)

    # Unpack parameters
    params = np.asarray(params).reshape(7, 5)
    A, B, C, D, E, F, G = params  # each shape (5,)

    # Expand to (n,5)
    A = A[None, :]
    B = B[None, :]
    C = C[None, :]
    D = D[None, :]
    E = E[None, :]
    F = F[None, :]
    G = G[None, :]

    # Compute terms
    term_self = B * np.power(p_safe, -C)
    term_comp = D * np.power(one_minus, -E)
    term_ent  = F * H            # broadcasting H => (n,1) to (n,5)
    term_int  = G * p_safe * one_minus

    # Sum up
    losses = A + term_self + term_comp + term_ent + term_int
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35 parameters to observed losses.

    Args:
      proportions: ndarray, shape (n_samples,5)
      loss_values: ndarray, shape (n_samples,5)

    Returns:
      best_params: ndarray, shape (35,)
    """
    p = np.atleast_2d(proportions).astype(float)
    y = np.atleast_2d(loss_values).astype(float)
    n, d = p.shape
    assert d == 5 and y.shape == (n,5)

    # Precompute entropy
    eps = 1e-8
    p_safe = np.clip(p, eps, 1.0)
    H = -np.sum(p_safe * np.log(p_safe), axis=1)

    # Initial parameter guess
    A0 = np.mean(y, axis=0)
    B0 = np.ones(5)
    C0 = np.ones(5) * 1.0
    D0 = np.ones(5)
    E0 = np.ones(5) * 1.0
    F0 = np.zeros(5)
    G0 = np.zeros(5)
    x0 = np.concatenate([A0, B0, C0, D0, E0, F0, G0])  # length 35

    # Bounds
    bounds_lower = []
    bounds_upper = []
    # A_i in [0, 10*mean]
    for Ai in A0:
        bounds_lower.append(0.0)
        bounds_upper.append(10.0 * Ai + 1e-6)
    # B,D in [1e-6, 10]
    for _ in range(5): bounds_lower.append(1e-6); bounds_upper.append(10.0)
    for _ in range(5): bounds_lower.append(-5.0); bounds_upper.append(5.0)   # C
    for _ in range(5): bounds_lower.append(1e-6); bounds_upper.append(10.0)  # D
    for _ in range(5): bounds_lower.append(-5.0); bounds_upper.append(5.0)   # E
    # F,G in [-10,10]
    for _ in range(5): bounds_lower.append(-10.0); bounds_upper.append(10.0)
    for _ in range(5): bounds_lower.append(-10.0); bounds_upper.append(10.0)

    bounds = (np.array(bounds_lower), np.array(bounds_upper))

    # Residual function for least_squares
    def residuals(x):
        pred = scaling_law_func(p, x)
        return (pred - y).ravel()

    # Multi‐start optimization
    best_x = x0.copy()
    best_cost = np.inf
    seeds = [0, 42, 123]
    for sd in seeds:
        rng = np.random.RandomState(sd)
        x_init = x0 + rng.randn(35) * 0.1
        res = least_squares(
            residuals,
            x_init,
            bounds=bounds,
            verbose=0,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=1000,
            method='trf'
        )
        if res.success and res.cost < best_cost:
            best_cost = res.cost
            best_x = res.x

    return best_x

# Expose expected parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END