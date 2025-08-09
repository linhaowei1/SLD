# EVOLVE-BLOCK-START
"""
Improved scaling law discovery for LLM finetuning scenarios.

We model:
    loss ≈ bias + exp(intercept + Σ_i w_i * log(x_i))
where x = [lr, bsz, data_size, non_embedding_param_size].

This captures a multiplicative power‐law via the exp‐linear term,
plus an additive floor (bias). We fit initial weights by linear
regression in log‐space, then refine via robust nonlinear least
squares (Huber loss) for stability.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predicts LM loss from hyperparameters via:
      loss_pred = bias + exp(intercept + w^T log(X))
    Inputs:
      data_points: array of shape (N,4)
      params:       array of shape (6,) = [bias, intercept, w_lr, w_bsz, w_data, w_param]
    Returns:
      preds: array of shape (N,)
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))  # (N,4)
    bias      = params[0]
    intercept = params[1]
    weights   = params[2:]                                        # (4,)
    # avoid log(0)
    eps = 1e-12
    X_clamped = np.maximum(X, eps)
    logX = np.log(X_clamped)                                      # (N,4)
    lin_term = intercept + logX.dot(weights)                      # (N,)
    # model output
    return bias + np.exp(lin_term)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the model:
      loss ≈ bias + exp(intercept + Σ_i w_i * log(x_i))
    by:
      1) Initial ridge‐regularized linear fit in log‐space to get
         intercept and weights.
      2) Bias floor initialized to half the minimum observed loss.
      3) Refinement via robust nonlinear least‐squares (Huber loss).
    Inputs:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)
    Returns:
      params: array of shape (6,) = [bias, intercept, w_lr, w_bsz, w_data, w_param]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))  # (N,4)
    y = np.asarray(loss_values, dtype=np.float64)                 # (N,)
    N, F = X.shape
    assert F == 4, "Expect 4 features: [lr, bsz, data_size, param_size]"

    # clamp to avoid log(0)
    eps = 1e-12
    X_clamped = np.maximum(X, eps)
    y_clamped = np.maximum(y, eps)

    # build log‐space design matrix for initial linear regression
    logX = np.log(X_clamped)               # (N,4)
    logy = np.log(y_clamped)               # (N,)

    # design matrix [1, logX]
    ones = np.ones((N, 1), dtype=np.float64)
    design = np.hstack((ones, logX))       # (N,5)

    # ridge regularization for stability (no penalty on intercept)
    lambda_reg = 1e-6
    I = np.eye(F+1, dtype=np.float64)
    I[0,0] = 0.0

    A = design.T.dot(design) + lambda_reg * I
    b = design.T.dot(logy)
    sol = np.linalg.solve(A, b)            # (5,)

    intercept0 = sol[0]
    weights0   = sol[1:]                   # (4,)

    # initialize bias floor to half the min observed loss (but at least eps)
    bias0 = max(y.min() * 0.5, eps)

    # initial parameter vector: [bias, intercept, w_lr, w_bsz, w_data, w_param]
    p0 = np.concatenate(([bias0, intercept0], weights0))

    # residual function for robust least squares
    def residuals(p):
        bias_p      = p[0]
        intercept_p = p[1]
        w_p         = p[2:]
        log_term = intercept_p + np.log(np.maximum(X, eps)).dot(w_p)
        y_pred = bias_p + np.exp(log_term)
        return y_pred - y

    # refine with Huber loss for robustness
    try:
        result = least_squares(
            residuals,
            p0,
            loss='huber',
            f_scale=1.0,
            max_nfev=1000,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8
        )
        p_opt = result.x if result.success else p0
    except Exception:
        p_opt = p0

    return p_opt
# EVOLVE-BLOCK-END