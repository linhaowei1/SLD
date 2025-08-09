# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Extended log‐linear scaling law with cross‐terms:
      log(loss) ≈ a 
                  + b·log(tokens) 
                  + c·log(params) 
                  + d·log(unique_tokens)
                  + e·[log(tokens)·log(params)]
                  + f·[log(tokens)·log(unique_tokens)]
    => loss = exp(log(loss))

    Inputs:
      data_points: array‐like of shape (N, 3) with columns [tokens, params, unique_tokens]
      params:      array‐like of length 6 [a, b, c, d, e, f]

    Returns:
      preds: shape (N,)
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.shape[1] != 3:
        raise ValueError(f"Expected data_points with 3 columns, got {X.shape[1]}")

    p = np.asarray(params, dtype=float).ravel()
    if p.size != 6:
        raise ValueError(f"Expected 6 parameters [a,b,c,d,e,f], got {p.size}")
    a, b, c, d, e, f = p

    # avoid log(0)
    eps = 1e-12
    log_toks = np.log(X[:, 0] + eps)
    log_params = np.log(X[:, 1] + eps)
    log_uniq = np.log(X[:, 2] + eps)

    # compute log‐prediction with cross terms
    log_pred = (
        a
        + b * log_toks
        + c * log_params
        + d * log_uniq
        + e * (log_toks * log_params)
        + f * (log_toks * log_uniq)
    )

    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the extended log‐linear scaling law by minimizing MSE in log‐space.
    Uses six parameters [a,b,c,d,e,f] as in scaling_law_func above.

    Returns:
      params_opt: array of 6 parameters [a, b, c, d, e, f]
    """
    X_raw = np.atleast_2d(np.asarray(data_points, dtype=float))
    y_raw = np.asarray(loss_values, dtype=float).ravel()

    # only positive losses for log‐domain fitting
    mask = y_raw > 0
    if not np.any(mask):
        raise ValueError("Need some positive loss values to fit model.")
    X = X_raw[mask]
    y = y_raw[mask]

    # logs with small epsilon
    eps = 1e-12
    log_toks = np.log(X[:, 0] + eps)
    log_params = np.log(X[:, 1] + eps)
    log_uniq = np.log(X[:, 2] + eps)
    y_log = np.log(y + eps)

    # design matrix with cross‐terms: [1, Lt, Lp, Lu, Lt*Lp, Lt*Lu]
    D = np.column_stack([
        np.ones_like(log_toks),
        log_toks,
        log_params,
        log_uniq,
        log_toks * log_params,
        log_toks * log_uniq
    ])

    # least‐squares init in log‐space
    beta_init, *_ = np.linalg.lstsq(D, y_log, rcond=None)

    # objective: mean squared error in log‐space
    def obj(p):
        resid = D.dot(p) - y_log
        return np.mean(resid**2)

    # refine via L-BFGS-B
    res = minimize(obj, beta_init, method="L-BFGS-B")
    if res.success:
        return res.x
    else:
        return beta_init
# EVOLVE-BLOCK-END