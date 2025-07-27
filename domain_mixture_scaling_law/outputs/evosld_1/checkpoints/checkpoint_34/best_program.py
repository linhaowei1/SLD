import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given 5-domain mixture proportions and a
    7-parameter power-law + entropy model per domain:

      L_i = A_i
            + B_i * (p_i + ε)^(-C_i)
            + D_i * H^E_i
            + F_i * (1 - p_i + ε)^G_i

    where H = -sum_j p_j * log(p_j) is the mixture entropy,
    and ε ensures numerical stability.
    """
    p = np.atleast_2d(proportions).astype(float)
    n_samples, d = p.shape
    assert d == 5, "Expected 5 domain proportions"
    eps = 1e-8

    # Clip proportions for stability
    p_safe = np.clip(p, eps, 1.0)
    one_minus_p = np.clip(1.0 - p, eps, 1.0)

    # Compute mixture entropy H for each sample (shape (n_samples, 1))
    H = -np.sum(p_safe * np.log(p_safe), axis=1, keepdims=True)

    # Unpack parameters: 7 per domain, total 35
    P = np.asarray(params, dtype=float).flatten()
    assert P.size == 35, f"Expected 35 parameters, got {P.size}"
    # Reshape to (5 domains, 7 params per domain)
    P = P.reshape(5, 7)
    A = P[:, 0][None, :]   # shape (1,5)
    B = P[:, 1][None, :]
    C = P[:, 2][None, :]
    D = P[:, 3][None, :]
    E = P[:, 4][None, :]
    F = P[:, 5][None, :]
    G = P[:, 6][None, :]

    # Compute each term
    term1 = B * np.power(p_safe, -C)        # power-law on p_i
    term2 = D * np.power(H, E)              # entropy-based term
    term3 = F * np.power(one_minus_p, G)     # complement power-law

    # Sum up to get per-domain losses
    losses = A + term1 + term2 + term3
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law parameters to observed losses using bounded
    least-squares with multi-start.

    Args:
        proportions: ndarray, shape (n_samples, 5)
        loss_values: ndarray,   shape (n_samples, 5)

    Returns:
        best_params: ndarray, shape (35,)
            Optimized parameters for the scaling law.
    """
    p = np.atleast_2d(proportions).astype(float)
    L_true = np.atleast_2d(loss_values).astype(float)
    n, d = p.shape
    assert d == 5 and L_true.shape == (n, 5), "Input shapes mismatch"

    # Initial guess based on data moments
    mean_L = np.mean(L_true, axis=0)
    A0 = mean_L
    B0 = np.ones(5)
    C0 = np.full(5, 0.5)
    D0 = np.ones(5)
    E0 = np.full(5, 0.5)
    F0 = np.ones(5)
    G0 = np.ones(5)
    x0 = np.concatenate([A0, B0, C0, D0, E0, F0, G0])  # length 35

    # Build bounds for each parameter
    lower = []
    upper = []
    for i in range(5):
        lower += [0.0,       # A_i
                  0.0,       # B_i
                  0.0,       # C_i
                  0.0,       # D_i
                  0.0,       # E_i
                  0.0,       # F_i
                  0.0]       # G_i
        upper += [10.0 * mean_L[i],  # A_i
                  10.0,              # B_i
                  5.0,               # C_i
                  10.0,              # D_i
                  5.0,               # E_i
                  10.0,              # F_i
                  5.0]               # G_i

    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    # Residual function for least_squares
    def _residuals(x):
        pred = scaling_law_func(p, x)
        return (pred - L_true).ravel()

    best_x = x0.copy()
    best_cost = np.inf

    # Multi-start to escape local minima
    for seed in [0, 1, 2]:
        rng = np.random.RandomState(seed)
        x_init = x0 + rng.randn(35) * 0.1
        res = least_squares(
            _residuals,
            x_init,
            bounds=(lower, upper),
            xtol=1e-9,
            ftol=1e-9,
            max_nfev=500
        )
        if res.success and res.cost < best_cost:
            best_cost = res.cost
            best_x = res.x

    return best_x

# Metadata
scaling_law_func.num_params = 35