# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict LM loss from hyperparameters via an enhanced power‐law model with
    interaction and squared log‐terms:
       log_loss ≈ p0
                 + p1*log(lr)
                 + p2*log(bsz)
                 + p3*log(data_size)
                 + p4*log(param_size)
                 + p5*[log(data_size) * log(param_size)]
                 + p6*[log(data_size)]^2
                 + p7*[log(param_size)]^2
       loss = exp(log_loss)
    params: [p0,...,p7]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))  # (N,4)
    eps = 1e-12
    X = np.maximum(X, eps)                                        # avoid log(0)
    logX = np.log(X)                                              # (N,4)
    log_lr, log_bsz = logX[:, 0], logX[:, 1]
    log_data, log_param = logX[:, 2], logX[:, 3]

    # additional basis functions
    cross_dp = log_data * log_param
    sq_data  = log_data**2
    sq_param = log_param**2

    # build design matrix: [1, log_lr, log_bsz, log_data, log_param,
    #                       cross_dp, sq_data, sq_param]
    N = X.shape[0]
    design = np.column_stack([
        np.ones(N, dtype=np.float64),
        log_lr,
        log_bsz,
        log_data,
        log_param,
        cross_dp,
        sq_data,
        sq_param
    ])  # (N,8)

    p = np.asarray(params, dtype=np.float64).ravel()             # (8,)
    log_pred = design.dot(p)                                     # (N,)

    # clip for numerical stability
    log_pred = np.clip(log_pred, -50.0, 50.0)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the enhanced power‐law model via ridge‐regularized linear regression
    in log‐space over an expanded basis [log, interaction, squares].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))  # (N,4)
    y = np.asarray(loss_values, dtype=np.float64)                 # (N,)

    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    logX = np.log(X)                                              # (N,4)
    logy = np.log(y)                                              # (N,)

    log_lr, log_bsz = logX[:, 0], logX[:, 1]
    log_data, log_param = logX[:, 2], logX[:, 3]

    cross_dp = log_data * log_param
    sq_data  = log_data**2
    sq_param = log_param**2

    N = X.shape[0]
    design = np.column_stack([
        np.ones(N, dtype=np.float64),
        log_lr,
        log_bsz,
        log_data,
        log_param,
        cross_dp,
        sq_data,
        sq_param
    ])  # (N,8)

    # ridge regularization (no penalty on intercept)
    D = design.shape[1]
    lam = 1e-3
    I = np.eye(D, dtype=np.float64)
    I[0, 0] = 0.0

    A = design.T.dot(design) + lam * I      # (8,8)
    b = design.T.dot(logy)                  # (8,)

    try:
        params = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # fallback to least squares if singular
        params, *_ = np.linalg.lstsq(design, logy, rcond=None)

    return params
# EVOLVE-BLOCK-END