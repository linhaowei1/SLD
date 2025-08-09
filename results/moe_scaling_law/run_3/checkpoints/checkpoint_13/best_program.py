import numpy as np

# EVOLVE-BLOCK-START
"""
Robust log–quadratic interaction scaling law:
  ln L = θ0 + θ1·lnP + θ2·lnE + θ3·(lnP·lnE) + θ4·(lnP)^2 + θ5·(lnE)^2
We fit by iteratively re‐weighted least squares (Huber) on ln(loss)
to reduce sensitivity to outliers while retaining a closed‐form solve.
"""
def scaling_law_func(data_points, params):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    # split and ensure positivity
    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)
    θ = np.asarray(params, dtype=float).ravel()
    if θ.size != 6:
        raise ValueError(f"Expected 6 parameters, got {θ.size}")
    θ0, θ1, θ2, θ3, θ4, θ5 = θ
    u = (
        θ0
        + θ1 * lnP
        + θ2 * lnE
        + θ3 * (lnP * lnE)
        + θ4 * (lnP ** 2)
        + θ5 * (lnE ** 2)
    )
    return np.exp(u)

def fit_scaling_law(data_points, loss_values):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()
    # ensure positive losses for log
    y_pos = np.clip(y, 1e-8, None)
    ln_y = np.log(y_pos)

    # features
    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)

    # design matrix: [1, lnP, lnE, lnP·lnE, (lnP)^2, (lnE)^2]
    M = np.vstack([
        np.ones_like(lnP),
        lnP,
        lnE,
        lnP * lnE,
        lnP**2,
        lnE**2
    ]).T  # shape (N,6)

    # initial ordinary least‐squares
    θ, *_ = np.linalg.lstsq(M, ln_y, rcond=None)

    # robustify via Huber IRLS (a few iterations)
    for _ in range(3):
        # residuals in log‐space
        r = M.dot(θ) - ln_y
        # robust scale estimate (MAD)
        mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-8
        delta = 1.345 * mad
        # Huber weights
        w = np.ones_like(r)
        mask = np.abs(r) > delta
        w[mask] = delta / np.abs(r[mask])
        # weighted least squares: solve (M^T W M) θ = M^T W ln_y
        Wm = w[:, None] * M            # shape (N,6)
        A = M.T.dot(Wm)                # (6,6)
        b = M.T.dot(w * ln_y)          # (6,)
        θ = np.linalg.lstsq(A, b, rcond=None)[0]

    return θ
# EVOLVE-BLOCK-END