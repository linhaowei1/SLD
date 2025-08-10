# EVOLVE-BLOCK-START
"""
Improved scaling law discovery for LLM finetuning scenarios.
Model: loss ≈ b + A * lr^{e_lr} * bsz^{e_bsz} * data_size^{e_D} * non_emb_p{e_N}
with A = exp(logA), fitted via bounded L-BFGS-B from multiple starts.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict lm loss given hyperparameters via a multiplicative power law plus bias.
    data_points: (N,4) array with columns [lr, bsz, data_size, non_embedding_param_size]
    params: (6,) array [b, logA, e_lr, e_bsz, e_data, e_model]
    Returns: (N,) array of predicted losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    lr   = X[:, 0]
    bsz  = X[:, 1]
    D    = X[:, 2]
    Np   = X[:, 3]
    b, logA, e_lr, e_bsz, e_D, e_N = params
    # Numerical stability epsilon
    eps = 1e-12
    # Compute log-term for stability: log_term = logA + e_lr*ln(lr) + ...
    log_term = (
        logA
        + e_lr  * np.log(lr  + eps)
        + e_bsz * np.log(bsz + eps)
        + e_D   * np.log(D    + eps)
        + e_N   * np.log(Np   + eps)
    )
    term = np.exp(log_term)
    return b + term

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter multiplicative scaling law to minimize MSE.
    Returns optimized params: [b, logA, e_lr, e_bsz, e_data, e_model]
    If multi-target loss_values given (shape N×T), returns array (T,6).
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float)
    # Ensure 2D output for possibly multi-target regression
    if y.ndim == 1:
        y2d = y[:, None]
    else:
        y2d = y
    T = y2d.shape[1]

    def _fit_single(y_vec):
        # Vector target y_vec of shape (N,)
        y_min, y_max = np.min(y_vec), np.max(y_vec)
        # Two reasonable initializations
        init1 = np.array([max(0.0, y_min * 0.5), np.log(max(y_max - y_min, 1e-3)),
                          -0.5, -0.5, -0.5, -0.5], dtype=float)
        init2 = np.array([0.0, np.log(max(y_max, 1e-3)),
                          -1.0, -1.0, -0.1, -0.1], dtype=float)
        # Bounds: bias b in [0, 2*max(y)], logA unbounded, exponents in [-5,5]
        bounds = [
            (0.0, max(y_max * 2, 1.0)),
            (None, None),
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-5.0, 5.0),
        ]
        best_params = None
        best_loss = np.inf

        # Objective: mean squared error
        def obj(p):
            pred = scaling_law_func(X, p)
            return np.mean((pred - y_vec) ** 2)

        # Try both initializations
        for init in (init1, init2):
            res = minimize(obj, init, method='L-BFGS-B', bounds=bounds)
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x

        # Fallback to init1 if all fails
        return best_params if best_params is not None else init1

    # Fit for each target dimension
    if T == 1:
        return _fit_single(y2d[:, 0])
    else:
        params_all = [ _fit_single(y2d[:, i]) for i in range(T) ]
        return np.vstack(params_all)
# EVOLVE-BLOCK-END