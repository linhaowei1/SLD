# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict lm loss from hyperparameters using a multiplicative power‐law:
      loss ≈ exp(intercept) * Π_i x_i**exponent_i

    Inputs:
      data_points: (N, F) array of positive hyperparameters
      params:     (P,) or (M, P) array of learned parameters,
                  where P = F + 1  (intercept + one exponent per feature)
    Returns:
      preds: shape (N,) if single output or (N, M) if M‐output
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    eps = 1e-12
    # log‐transform the features for a linear model in log‐space
    logX = np.log(X + eps)              # (N, F)

    theta = np.asarray(params, dtype=float)
    # support multi‐target: shape into (M, P)
    if theta.ndim == 1:
        theta = theta[None, :]
    M, P = theta.shape
    N, F = X.shape
    if P != F + 1:
        raise ValueError(f"Expected param length {F+1}, got {P}")

    # design matrix Z = [1, log(x1), log(x2), ..., log(xF)]
    Z = np.concatenate([np.ones((N, 1)), logX], axis=1)  # (N, P)

    # linear prediction in log‐space, then exponentiate
    pred_log = Z.dot(theta.T)                            # (N, M)
    pred = np.exp(pred_log)                              # (N, M)

    # flatten if only one target
    return pred.ravel() if M == 1 else pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the intercept + exponents in log‐space via (ridge‐regularized)
    linear regression:
      log(loss) ≈ intercept + Σ_i exponent_i * log(x_i)

    Returns:
      params: shape (P,) or (T, P) if multi‐target,
              where P = F + 1
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float)
    N, F = X.shape
    eps = 1e-12

    # build design matrix in log‐space
    logX = np.log(X + eps)              # (N, F)
    Z = np.concatenate([np.ones((N, 1)), logX], axis=1)  # (N, P)
    P = F + 1

    # small ridge penalty for numeric stability (no penalty on intercept)
    lambda_reg = 1e-6
    reg = np.eye(P)
    reg[0, 0] = 0
    A = Z.T.dot(Z) + lambda_reg * reg   # (P, P)

    # support multi‐target losses
    if y.ndim == 1:
        y2d = y[:, None]
    else:
        y2d = y
    T = y2d.shape[1]

    params = np.zeros((T, P), dtype=float)
    for t in range(T):
        logy = np.log(y2d[:, t] + eps)  # (N,)
        b = Z.T.dot(logy)                # (P,)
        params[t] = np.linalg.solve(A, b)

    # return (P,) when single target, else (T, P)
    return params[0] if T == 1 else params
# EVOLVE-BLOCK-END