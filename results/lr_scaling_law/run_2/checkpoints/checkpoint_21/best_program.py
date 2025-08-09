import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Enhanced scaling law function supporting:
      (a) pure power‐law with 5 parameters: [bias, w_lr, w_bsz, w_data, w_param]
      (b) cross‐quadratic law with 11 parameters:
          [bias,
           w1..w4 (linear in log),
           q1..q4 (quadratic in log),
           c1=log(lr)*log(bsz),
           c2=log(data)*log(param)]
    """
    X = np.asarray(data_points, dtype=float)
    eps = 1e-12
    X = np.maximum(X, eps)         # avoid log(0)
    logX = np.log(X)               # (N,4)

    theta = np.asarray(params, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    M, P = theta.shape
    N, F = logX.shape

    # Build design matrix Z based on parameter length
    if P == F + 1:
        # pure power‐law: bias + linear terms
        Z = np.concatenate([np.ones((N, 1)), logX], axis=1)
    elif P == 11:
        # cross‐quadratic law: bias + linear + quadratic + two cross terms
        lr   = logX[:, 0:1]   # log(lr)
        bsz  = logX[:, 1:2]   # log(bsz)
        data = logX[:, 2:3]   # log(data_size)
        prm  = logX[:, 3:4]   # log(param_size)
        lin   = np.concatenate([lr, bsz, data, prm], axis=1)
        quad  = lin**2
        cross = np.concatenate([lr * bsz, data * prm], axis=1)
        Z     = np.concatenate([np.ones((N, 1)), lin, quad, cross], axis=1)
    else:
        raise ValueError(f"Unsupported params length {P}, expected {F+1} or 11")

    # Linear model in log-space → exponentiate to get loss
    log_pred = Z.dot(theta.T)      # (N, M)
    pred     = np.exp(log_pred)    # (N, M)

    return pred.ravel() if M == 1 else pred

def fit_scaling_law(data_points, loss_values):
    """
    Fits the cross‐quadratic scaling law:
      log(loss) ≈ bias
                  + Σ_i w_i·log(x_i)
                  + Σ_i q_i·[log(x_i)]^2
                  + c1·log(lr)·log(bsz)
                  + c2·log(data_size)·log(param_size)

    Solves a ridge‐regularized linear system in log-space.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    eps = 1e-12

    # Clip to avoid log(0)
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    N, F = X.shape
    logX = np.log(X)         # (N,4)
    logy = np.log(y)         # (N,)

    # Split features
    lr   = logX[:, 0:1]
    bsz  = logX[:, 1:2]
    data = logX[:, 2:3]
    prm  = logX[:, 3:4]

    # Build design matrix Z: bias + linear(4) + quad(4) + cross(2) = 11 cols
    lin   = np.concatenate([lr, bsz, data, prm], axis=1)     # (N,4)
    quad  = lin**2                                             # (N,4)
    cross = np.concatenate([lr * bsz, data * prm], axis=1)    # (N,2)
    Z     = np.concatenate([np.ones((N,1)), lin, quad, cross], axis=1)  # (N,11)

    # Ridge regularization (no penalty on bias)
    P = Z.shape[1]
    lambda_reg = 1e-2
    reg = np.eye(P)
    reg[0, 0] = 0.0

    A = Z.T.dot(Z) + lambda_reg * reg
    b = Z.T.dot(logy)

    # Solve for parameters in log-space
    params = np.linalg.solve(A, b)  # (11,)

    return params

# EVOLVE-BLOCK-END