import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict language‐model loss from hyperparameters via a simplified
    2nd‐order log‐polynomial with one cross‐interaction.

    Model form in the log‐domain:
      log(y_pred) = p0
                  + p1*L_lr   + p2*L_bsz   + p3*L_data   + p4*L_param
                  + p5*L_lr^2 + p6*L_bsz^2 + p7*L_data^2 + p8*L_param^2
                  + p9*(L_data * L_param)

    where L_x = log(x), and x = [lr, bsz, data_size, non_embedding_param_size].
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected input with 4 features, got {F}")
    p = np.asarray(params, dtype=float).ravel()
    P_expected = 10
    if p.shape[0] != P_expected:
        raise ValueError(f"Expected {P_expected} parameters, got {p.shape[0]}")

    # avoid log(0)
    X_clipped = np.maximum(X, 1e-12)
    logX = np.log(X_clipped)
    L_lr    = logX[:, 0]
    L_bsz   = logX[:, 1]
    L_data  = logX[:, 2]
    L_param = logX[:, 3]

    # build design matrix Phi (N x 10)
    # [1, L_lr, L_bsz, L_data, L_param, L_lr^2, L_bsz^2, L_data^2, L_param^2, L_data*L_param]
    Phi = np.stack([
        np.ones(N),
        L_lr, L_bsz, L_data, L_param,
        L_lr**2, L_bsz**2, L_data**2, L_param**2,
        L_data * L_param
    ], axis=1)

    log_pred = Phi.dot(p)       # shape (N,)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the simplified 2nd‐order log‐polynomial scaling law via
    ridge‐regularized closed‐form regression.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    y = np.asarray(loss_values, dtype=float).ravel()

    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected data_points with 4 features, got {F}")
    if y.shape[0] != N:
        raise ValueError("Number of data points and loss values must match")

    # avoid log(0)
    X_clipped = np.maximum(X, 1e-12)
    y_clipped = np.maximum(y, 1e-12)

    logX = np.log(X_clipped)
    logy = np.log(y_clipped)

    L_lr    = logX[:, 0]
    L_bsz   = logX[:, 1]
    L_data  = logX[:, 2]
    L_param = logX[:, 3]

    # build design matrix Phi (N x 10)
    Phi = np.stack([
        np.ones(N),
        L_lr, L_bsz, L_data, L_param,
        L_lr**2, L_bsz**2, L_data**2, L_param**2,
        L_data * L_param
    ], axis=1)

    # ridge‐regularized normal equations
    P = Phi.shape[1]   # should be 10
    ridge = 1e-6
    A = Phi.T.dot(Phi)
    # apply ridge on all terms except intercept
    diag_idx = np.arange(1, P)
    A[diag_idx, diag_idx] += ridge
    b = Phi.T.dot(logy)

    # solve for parameters
    params = np.linalg.solve(A, b)
    return params