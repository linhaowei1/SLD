import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Log‐linear scaling law:
      log(loss) ≈ a + b·log(tokens) + c·log(params) + d·log(unique_tokens)
    => loss = exp(a) * tokens^b * params^c * unique_tokens^d

    Inputs:
      data_points: array‐like of shape (N, 3) with columns [tokens, params, unique_tokens]
      params:      array of length 4 (or shape (M,4) for M independent outputs)
                   [a, b, c, d]

    Returns:
      preds: shape (N,) (or (N, M) if params is (M,4))
    """
    X = np.atleast_2d(data_points).astype(float)
    # avoid log(0)
    X_log = np.log(X + 1e-12)

    p = np.asarray(params, dtype=float)
    # ensure 2D: (M,4)
    if p.ndim == 1:
        p2 = p[np.newaxis, :]
    else:
        p2 = p.copy()
    M, P = p2.shape
    if P != 4:
        raise ValueError(f"scaling_law_func expects 4 parameters (a,b,c,d), got {P}")

    # unpack
    a = p2[:, 0]    # shape (M,)
    b = p2[:, 1]
    c = p2[:, 2]
    d = p2[:, 3]

    # compute log‐predictions: shape (N, M)
    # X_log[:,0] = log(tokens), X_log[:,1] = log(params), X_log[:,2] = log(unique_tokens)
    log_pred = (
        a[np.newaxis, :]
        + b[np.newaxis, :] * X_log[:, 0:1]
        + c[np.newaxis, :] * X_log[:, 1:2]
        + d[np.newaxis, :] * X_log[:, 2:3]
    )
    pred = np.exp(log_pred)

    return pred[:, 0] if M == 1 else pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the log‐linear scaling law by minimizing mean squared error in log‐space.

    Returns 4 parameters [a, b, c, d] such that
      log(loss) ≈ a + b·log(tokens) + c·log(params) + d·log(unique_tokens)
    """
    X_raw = np.atleast_2d(data_points).astype(float)  # (N,3)
    y_raw = np.asarray(loss_values, dtype=float).reshape(-1)  # (N,)

    # keep only strictly positive losses for log
    mask = y_raw > 0
    X = X_raw[mask]
    y = y_raw[mask]

    # take logs with small offset
    X_log = np.log(X + 1e-12)  # (N,3)
    y_log = np.log(y + 1e-12)  # (N,)

    N = X_log.shape[0]
    # Design matrix for linear regression in log‐space: [1, log(tokens), log(params), log(unique)]
    D = np.concatenate([np.ones((N, 1)), X_log], axis=1)  # (N,4)

    # initial least‐squares solution in log‐space
    beta_init, *_ = np.linalg.lstsq(D, y_log, rcond=None)  # shape (4,)

    # objective: MSE in log domain
    def objective(p):
        resid = D.dot(p) - y_log
        return np.mean(resid * resid)

    # boundless optimization (L-BFGS-B)
    res = minimize(objective, beta_init, method="L-BFGS-B")
    params_opt = res.x if res.success else beta_init

    return params_opt