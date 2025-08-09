import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Supports either
      (a) a pure power‐law:    P = F+1  params  = [intercept, a1, …, aF]
      (b) a quadratic log‐law: P = 2F+1 params = [intercept, a1…aF, b1…bF]
    where F = number of hyperparameters (here 4).
    """
    X = np.asarray(data_points, dtype=float)
    eps = 1e-12
    N, F = X.shape

    # log‐transform
    logX = np.log(X + eps)               # (N, F)

    theta = np.asarray(params, dtype=float)
    # promote to 2D for multi‐target
    if theta.ndim == 1:
        theta = theta[None, :]
    M, P = theta.shape

    # choose design matrix by parameter length
    if P == F + 1:
        # pure power‐law
        Z = np.concatenate([np.ones((N, 1)), logX], axis=1)           # (N, F+1)
    elif P == 2*F + 1:
        # quadratic in log‐space
        Z = np.concatenate([np.ones((N, 1)), logX, logX**2], axis=1)  # (N, 1+F+F)
    else:
        raise ValueError(f"Expected params of length {F+1} or {2*F+1}, got {P}")

    # linear model in log‐space → exponentiate
    pred_log = Z.dot(theta.T)    # (N, M)
    pred     = np.exp(pred_log)  # (N, M)

    return pred.ravel() if M == 1 else pred


def fit_scaling_law(data_points, loss_values):
    """
    Always fits the quadratic log‐law:
       log(loss) ≈ intercept + Σ a_i·log(x_i) + Σ b_i·[log(x_i)]^2
    via ridge‐regularized least squares in log‐space.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    N, F = X.shape
    eps = 1e-12

    # log‐space features
    logX = np.log(X + eps)                         # (N, F)
    Z    = np.concatenate([np.ones((N,1)), 
                           logX, 
                           logX**2], axis=1)      # (N, 1 + F + F)
    P    = Z.shape[1]

    # small ridge penalty (no penalty on intercept)
    lambda_reg = 1e-3
    reg        = np.eye(P)
    reg[0,0]   = 0
    A          = Z.T.dot(Z) + lambda_reg*reg       # (P, P)

    # support multi‐target y
    if y.ndim == 1:
        y2d = y[:,None]
    else:
        y2d = y
    T = y2d.shape[1]

    params = np.zeros((T, P), dtype=float)
    for t in range(T):
        logy       = np.log(y2d[:,t] + eps)        # (N,)
        b          = Z.T.dot(logy)                 # (P,)
        params[t]  = np.linalg.solve(A, b)

    return params[0] if T == 1 else params
# EVOLVE-BLOCK-END