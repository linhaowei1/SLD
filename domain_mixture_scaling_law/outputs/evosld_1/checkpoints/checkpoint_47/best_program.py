import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
"""
Refined Domain Mixture Scaling Law for LLM Training Scenarios

We model each domain's loss L_i as a 7‐parameter function per domain:
  L_i = A_i
        + B_i * (p_i + eps)^(−C_i)
        + D_i * (1 − p_i + eps)^(−E_i)
        + F_i * H
        + G_i * p_i * (1 − p_i)

where:
  - A_i          : base offset
  - B_i, C_i     : self‐proportion inverse power law
  - D_i, E_i     : complement inverse power law
  - F_i          : global mixture entropy coefficient
  - G_i          : local interaction coefficient
  - H = −∑_j p_j log(p_j) : mixture entropy
This uses exactly 7 parameters per domain (35 total).
"""

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given mixture proportions and parameters.

    Args:
        proportions: array-like, shape (n_samples, 5)
            Domain mixture proportions, each row sums to 1.
        params: array-like, shape (35,)
            Parameters stacked as [A_1..A_5, B_1..B_5, C_1..C_5,
                                  D_1..D_5, E_1..E_5, F_1..F_5, G_1..G_5].

    Returns:
        losses: ndarray, shape (n_samples, 5)
            Predicted loss for each domain and sample.
    """
    p = np.atleast_2d(proportions).astype(float)
    n, d = p.shape
    assert d == 5, "Expected 5 domain proportions"

    # numerical stability
    eps = 1e-8
    p_safe = np.clip(p, eps, 1.0)
    one_minus = np.clip(1.0 - p_safe, eps, 1.0)

    # global mixture entropy H (shape n x 1)
    H = -np.sum(p_safe * np.log(p_safe), axis=1, keepdims=True)

    # unpack parameters into 7 × 5 matrix
    params = np.asarray(params).flatten()
    assert params.size == 35, "Expected 35 parameters"
    P = params.reshape(7, 5)
    A, B, C, D, E, F, G = [row[None, :] for row in P]

    # compute terms
    term_self  = B * np.power(p_safe, -C)
    term_comp  = D * np.power(one_minus, -E)
    term_ent   = F * H
    term_inter = G * (p_safe * one_minus)

    losses = A + term_self + term_comp + term_ent + term_inter
    return losses


def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law parameters to observed losses.

    Args:
        proportions: ndarray, shape (n_samples, 5)
        loss_values: ndarray, shape (n_samples, 5)

    Returns:
        best_params: ndarray, shape (35,)
            Optimized parameters for the scaling law.
    """
    p = np.atleast_2d(proportions).astype(float)
    L = np.atleast_2d(loss_values).astype(float)
    n, d = p.shape
    assert d == 5 and L.shape == (n, 5)

    # initial guess
    A0 = np.maximum(np.mean(L, axis=0), 1e-3)
    B0 = np.ones(5)
    C0 = np.full(5, 1.0)
    D0 = np.ones(5)
    E0 = np.full(5, 1.0)
    F0 = np.zeros(5)
    G0 = np.zeros(5)
    x0 = np.concatenate([A0, B0, C0, D0, E0, F0, G0])

    # parameter bounds
    bounds = []
    maxL = np.max(L, axis=0)
    for i in range(5):
        bounds.append((0.0, maxL[i] * 10.0))   # A_i ≥0
    for _ in range(5):
        bounds.append((0.0, 10.0))             # B_i ≥0
    for _ in range(5):
        bounds.append((0.0, 5.0))              # C_i ≥0
    for _ in range(5):
        bounds.append((0.0, 10.0))             # D_i ≥0
    for _ in range(5):
        bounds.append((0.0, 5.0))              # E_i ≥0
    for _ in range(5):
        bounds.append((-10.0, 10.0))           # F_i (entropy coeff)
    for _ in range(5):
        bounds.append((-10.0, 10.0))           # G_i (interaction coeff)

    # objective: mean squared error
    def objective(x):
        pred = scaling_law_func(p, x)
        return np.mean((pred - L) ** 2)

    # multi-start optimization to avoid local minima
    best_x = x0.copy()
    best_val = objective(x0)
    rng = np.random.RandomState(42)
    for trial in range(5):
        x_init = x0 + rng.randn(35) * 0.1
        res = minimize(objective, x_init, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    return best_x

# attach metadata
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END