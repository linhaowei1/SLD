import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(proportions, params):
    """
    Simplified domain–mixture scaling law:
      For each domain i:
        Li = A_i
             + B_i * p_i^(–C_i)
             + D_i * (1 – p_i)^(E_i)
             + G_i * p_i * (1 – p_i)
             + F_i * H

      where
        p_i = proportion of domain i,
        H   = –∑_j p_j · log(p_j) (mixture entropy),
      and params is a flat array of length 35 = 5 domains × 7 parameters:
        [A,B,C,D,E,F,G] for each domain.
    """
    P = np.atleast_2d(proportions).astype(float)
    n_samples, n_dom = P.shape
    assert n_dom == 5, "Expected 5 domains"

    flat = np.asarray(params, dtype=float).ravel()
    if flat.size < 35:
        flat = np.pad(flat, (0, 35 - flat.size), 'constant')
    dom_p = flat[:35].reshape(5, 7)

    # clip proportions to avoid log(0) or division by zero
    Pc = np.clip(P, 1e-8, 1 - 1e-8)
    # mixture entropy H for each sample
    H = -np.sum(Pc * np.log(Pc), axis=1)  # shape (n_samples,)

    # unpack parameters for broadcasting
    A = dom_p[:, 0][None, :]  # shape (1,5)
    B = dom_p[:, 1][None, :]
    C = dom_p[:, 2][None, :]
    D = dom_p[:, 3][None, :]
    E = dom_p[:, 4][None, :]
    F = dom_p[:, 5][None, :]  # coefficient for entropy
    G = dom_p[:, 6][None, :]  # coefficient for p*(1-p) term

    p = Pc                # shape (n_samples,5)
    q = 1.0 - Pc          # shape (n_samples,5)
    Hmat = H[:, None]     # shape (n_samples,1)

    term_inv = B * np.power(p, -C)
    term_q   = D * np.power(q, E)
    term_mix = G * p * q
    term_ent = F * Hmat

    return A + term_inv + term_q + term_mix + term_ent


def fit_scaling_law(proportions, loss_values):
    """
    Fit the per-domain scaling law parameters by minimizing MSE
    with L-BFGS-B multi-start.
    Returns a flat array of length 35.
    """
    P = np.atleast_2d(proportions).astype(float)
    L = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = P.shape
    assert n_dom == 5 and L.shape == (n_samples, 5)

    # initialize: A_i = mean loss, B_i=D_i=1, C_i=E_i=1, F_i=0, G_i=0
    means = np.mean(L, axis=0)
    init = np.zeros((5, 7), dtype=float)
    init[:, 0] = means   # A_i
    init[:, 1] = 1.0     # B_i
    init[:, 2] = 1.0     # C_i
    init[:, 3] = 1.0     # D_i
    init[:, 4] = 1.0     # E_i
    init[:, 5] = 0.0     # F_i (entropy coef)
    init[:, 6] = 0.0     # G_i (mix-term coef)
    x0 = init.ravel()

    # bounds for stability
    bounds = []
    for _ in range(5):
        bounds += [
            (None, None),    # A_i
            (1e-6, None),    # B_i >0
            (0.1, 5.0),      # C_i
            (1e-6, None),    # D_i >0
            (0.1, 5.0),      # E_i
            (-1.0, 1.0),     # F_i
            (-1.0, 1.0)      # G_i
        ]

    def objective(x):
        pred = scaling_law_func(P, x)
        return np.mean((pred - L) ** 2)

    best_x = x0.copy()
    best_loss = objective(best_x)

    # multi-start with small perturbations
    for seed in range(3):
        rng = np.random.RandomState(seed)
        trial = x0 + 0.05 * rng.randn(x0.size)
        res = minimize(
            objective, trial,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x

    return best_x

# annotate parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END