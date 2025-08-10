import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict LM loss from hyperparameters using a weighted,
    regularized 2nd‐order log‐polynomial scaling law with one key interaction.

    Model in the log‐domain:
      log(y_pred) = p0
                   + p1*L_lr    + p2*L_bsz    + p3*L_data    + p4*L_param
                   + p5*L_lr^2  + p6*L_bsz^2  + p7*L_data^2  + p8*L_param^2
                   + p9*(L_data * L_param)

    where L_x = log(x), x = [lr, bsz, data_size, non_embedding_param_size].

    Args:
      data_points: array‐like of shape (N,4)
                   columns = [lr, bsz, data_size, non_embedding_param_size]
      params:      array of length 10:
                   [p0,
                    p1_lr, p2_bsz, p3_data, p4_param,
                    p5_lr2, p6_bsz2, p7_data2, p8_param2,
                    p9_data×param]

    Returns:
      y_pred: numpy array of shape (N,) of predicted LM losses.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.shape[1] != 4:
        raise ValueError(f"Expected input with 4 features, got {X.shape[1]}")
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 10:
        raise ValueError(f"Expected 10 parameters, got {p.size}")

    # safe log
    logX = np.log(np.maximum(X, 1e-12))
    L_lr, L_bsz, L_data, L_param = logX.T

    # design matrix: intercept, linear, quadratic, one cross-term
    Phi = np.column_stack([
        np.ones_like(L_lr),
        L_lr, L_bsz, L_data, L_param,
        L_lr**2, L_bsz**2, L_data**2, L_param**2,
        L_data * L_param
    ])

    log_pred = Phi.dot(p)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 2nd‐order log‐polynomial scaling law via weighted,
    differential‐ridge regression in closed form.

    We weight samples to emphasize large‐scale configurations
    (higher data_size and model_size), and apply light ridge on
    linear terms, moderate on quadratic, stronger on the data×model cross-term.

    Args:
      data_points: array‐like of shape (N,4)
      loss_values: array‐like of shape (N,)

    Returns:
      params: numpy array of length 10 of fitted coefficients.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float).ravel()
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if X.shape[1] != 4:
        raise ValueError(f"Expected data_points with 4 features, got {X.shape[1]}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of data points and loss values must match")

    # safe floor to avoid log issues
    X_safe = np.maximum(X, 1e-12)
    y_safe = np.maximum(y, 1e-12)

    # log-transform
    logX = np.log(X_safe)
    logy = np.log(y_safe)
    L_lr, L_bsz, L_data, L_param = logX.T

    # build design matrix
    Phi = np.column_stack([
        np.ones_like(L_lr),
        L_lr, L_bsz, L_data, L_param,
        L_lr**2, L_bsz**2, L_data**2, L_param**2,
        L_data * L_param
    ])

    # weight samples by scale: up-weight larger (data_size + model_size)
    scale_signal = L_data + L_param
    median_s = np.median(scale_signal)
    w = np.exp(0.5 * (scale_signal - median_s))
    W_sqrt = np.sqrt(w)[:, None]

    # weighted normal equations
    Phi_w = Phi * W_sqrt         # each row scaled by sqrt(w_i)
    y_w   = logy * W_sqrt.ravel()
    A = Phi_w.T.dot(Phi_w)
    b = Phi_w.T.dot(y_w)

    # differential ridge penalties
    # intercept idx=0 (no penalty)
    # linear idx 1-4, quadratic idx 5-8, cross-term idx 9
    ridge_lin   = 1e-6
    ridge_quad  = 1e-4
    ridge_cross = 1e-2

    for i in range(1, 5):
        A[i, i] += ridge_lin
    for i in range(5, 9):
        A[i, i] += ridge_quad
    A[9, 9] += ridge_cross

    # solve for parameters
    params = np.linalg.solve(A, b)
    return params