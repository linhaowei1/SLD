import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Improved log‐linear scaling law with pairwise interaction terms:
      log(loss) ≈ α 
                + β·log(tokens) 
                + γ·log(params) 
                + δ·log(unique_tokens)
                + ε·[log(tokens)·log(params)]
                + ζ·[log(tokens)·log(unique_tokens)]

    => loss = exp(log_pred)

    Inputs:
      data_points: array‐like of shape (N, 3) with columns 
                   [tokens, params, unique_tokens]
      params:      array‐like of length 6: 
                   [α, β, γ, δ, ε, ζ]
    Returns:
      preds:       np.ndarray of shape (N,) with predicted loss
    """
    X = np.atleast_2d(data_points).astype(float)
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")

    p = np.asarray(params, dtype=float).ravel()
    if p.size != 6:
        raise ValueError("params must have length 6: [α, β, γ, δ, ε, ζ]")

    # avoid log(0)
    X_log = np.log(np.maximum(X, 1e-12))
    t = X_log[:, 0]
    m = X_log[:, 1]
    u = X_log[:, 2]

    α, β, γ, δ, ε, ζ = p

    # compute log‐prediction with interactions
    log_pred = (
        α
        + β * t
        + γ * m
        + δ * u
        + ε * (t * m)
        + ζ * (t * u)
    )

    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the improved scaling law by minimizing mean squared error in log‐space.
    Returns optimized parameters [α, β, γ, δ, ε, ζ].
    """
    X_raw = np.atleast_2d(data_points).astype(float)
    y_raw = np.asarray(loss_values, dtype=float).reshape(-1)

    # only positive losses for log‐space fitting
    mask = y_raw > 0
    X = X_raw[mask]
    y = y_raw[mask]

    # logs with small offset
    X_log = np.log(np.maximum(X, 1e-12))
    y_log = np.log(y)

    t = X_log[:, 0]
    m = X_log[:, 1]
    u = X_log[:, 2]

    # construct design matrix with interactions: [1, t, m, u, t*m, t*u]
    D = np.stack([np.ones_like(t), t, m, u, t * m, t * u], axis=1)

    # initial least-squares solution in log‐space
    beta_init, *_ = np.linalg.lstsq(D, y_log, rcond=None)
    beta_init = beta_init.ravel()  # shape (6,)

    # objective: MSE in log‐space
    def objective(p):
        resid = D.dot(p) - y_log
        return np.mean(resid * resid)

    # refine via L-BFGS-B
    res = minimize(
        objective,
        beta_init,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-12}
    )

    if res.success and res.x.shape == beta_init.shape:
        return res.x
    else:
        # fallback to linear solution
        return beta_init

# EVOLVE-BLOCK-END