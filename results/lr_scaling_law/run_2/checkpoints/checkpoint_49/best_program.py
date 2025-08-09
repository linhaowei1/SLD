import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Full quadratic log‐law with cross‐terms:
      log(loss) ≈ θ0
                 + Σ_i θ_i · log(x_i)
                 + Σ_i θ_{i+4} · [log(x_i)]^2
                 + Σ_{i<j} θ_{9 + k} · log(x_i)·log(x_j)

    where x = [lr, bsz, data_size, non_embedding_param_size].
    Total parameters P = 1 + 4 + 4 + 6 = 15.
    """
    X = np.asarray(data_points, dtype=np.float64)
    # avoid log(0)
    eps = 1e-12
    X = np.maximum(X, eps)
    logX = np.log(X)  # shape (N,4)

    # unpack log‐features
    lr   = logX[:, 0]
    bsz  = logX[:, 1]
    ds   = logX[:, 2]
    ps   = logX[:, 3]
    N = logX.shape[0]

    # build design matrix Z: [1,
    #                         lr, bsz, ds, ps,
    #                         lr^2, bsz^2, ds^2, ps^2,
    #                         lr*bsz, lr*ds, lr*ps,
    #                         bsz*ds, bsz*ps,
    #                         ds*ps]
    Z = np.column_stack([
        np.ones(N, dtype=np.float64),
        lr, bsz, ds, ps,
        lr**2, bsz**2, ds**2, ps**2,
        lr * bsz, lr * ds, lr * ps,
        bsz * ds, bsz * ps,
        ds  * ps
    ])

    theta = np.asarray(params, dtype=np.float64).ravel()
    if theta.size != Z.shape[1]:
        raise ValueError(f"Expected {Z.shape[1]} parameters, got {theta.size}")

    # predict in log‐space, clamp for numeric stability
    log_pred = Z.dot(theta)
    log_pred = np.clip(log_pred, -50.0, 50.0)

    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the full quadratic log‐law with cross‐terms via ridge‐regularized
    weighted least squares in log‐space.

    We weight each sample by 1 + normalized( log(data_size) + log(param_size) )
    to emphasize large‐scale configurations.
    """
    X = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    logX = np.log(X)   # (N,4)
    logy = np.log(y)   # (N,)

    # unpack log‐features
    lr   = logX[:, 0]
    bsz  = logX[:, 1]
    ds   = logX[:, 2]
    ps   = logX[:, 3]
    N = logX.shape[0]

    # build design matrix Z as above
    Z = np.column_stack([
        np.ones(N, dtype=np.float64),
        lr, bsz, ds, ps,
        lr**2, bsz**2, ds**2, ps**2,
        lr * bsz, lr * ds, lr * ps,
        bsz * ds, bsz * ps,
        ds  * ps
    ])  # shape (N,15)

    # sample weights to emphasize large‐scale points
    log_scale = ds + ps
    m, M = log_scale.min(), log_scale.max()
    if M > m:
        w_scale = (log_scale - m) / (M - m)
    else:
        w_scale = np.zeros_like(log_scale)
    sample_w = 1.0 + w_scale            # in [1,2]
    sw_sqrt = np.sqrt(sample_w)

    # apply weights to design matrix and target
    Zw = Z * sw_sqrt[:, None]
    yw = logy * sw_sqrt

    # ridge regularization (no penalty on intercept)
    P = Z.shape[1]
    lambda_reg = 1e-3
    reg = np.eye(P, dtype=np.float64)
    reg[0, 0] = 0.0

    A = Zw.T.dot(Zw) + lambda_reg * reg  # (15,15)
    b = Zw.T.dot(yw)                     # (15,)

    theta = np.linalg.solve(A, b)       # (15,)
    return theta
# EVOLVE-BLOCK-END