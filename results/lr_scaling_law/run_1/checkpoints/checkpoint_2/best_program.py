# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(data_points, params):
    """
    Log‐linear scaling law:
      Loss = exp( θ0 + θ1*log(lr) + θ2*log(bsz)
                  + θ3*log(data_size) + θ4*log(non_embedding_param_size) )
    data_points: (N,4) array [lr, bsz, data_size, non_embed_param_size]
    params:      length‐5 array [θ0, θ1, θ2, θ3, θ4]
    returns     length‐N array of predicted losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    if X.shape[1] != 4:
        raise ValueError(f"scaling_law_func expects 4 features, got {X.shape[1]}")
    theta = np.asarray(params, dtype=np.float64).ravel()
    if theta.size != 5:
        raise ValueError(f"scaling_law_func expects params of length 5, got {theta.size}")
    # Avoid log(0)
    eps = 1e-12
    logs = np.log(X + eps)            # shape (N,4)
    log_pred = theta[0] + logs.dot(theta[1:])  # shape (N,)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the 5 parameters [θ0…θ4] by least‐squares on log(Loss).
    Solves:  log(y) ≃ θ0 + θ1*log(lr) + … + θ4*log(param_size)
    Returns: length‐5 array θ
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64).ravel()
    if X.shape[1] != 4:
        raise ValueError(f"fit_scaling_law expects 4 features, got {X.shape[1]}")
    # Build design matrix A = [1, log(lr), log(bsz), log(data_size), log(param_size)]
    eps = 1e-12
    Z = np.log(X + eps)                           # (N,4)
    A = np.concatenate((np.ones((Z.shape[0],1)), Z), axis=1)  # (N,5)
    y_log = np.log(y + eps)                       # (N,)
    # Closed‐form LS solution for θ
    theta, *_ = np.linalg.lstsq(A, y_log, rcond=None)
    return theta
# EVOLVE-BLOCK-END