# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models

This version fits a 6-parameter log-quadratic model in log-space via
ordinary least squares. It captures linear, quadratic, and interaction
effects of number of experts and total parameter count on validation loss.

Model form:
    ne = num_experts + eps
    np = (total_parameter_count / 1e6) + eps
    x  = log(ne)
    y  = log(np)

    log_loss ≈ a
               + b * x
               + c * y
               + d * x^2
               + e * y^2
               + f * x * y

    loss = exp(log_loss)

This uses exactly 6 parameters: [a, b, c, d, e, f].
Fitting is done with a single analytic least-squares solve.
"""
import numpy as np

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict validation loss given MoE configuration.

    Args:
        num_experts: array-like of shape (N,), number of experts per model.
        total_parameter_count: array-like of shape (N,), total model parameters.
        params: array-like of length 6: [a, b, c, d, e, f].

    Returns:
        loss_pred: ndarray of shape (N,), predicted loss values.
    """
    # Unpack parameters
    a, b, c, d, e, f = params

    # Convert inputs to arrays
    ne = np.asarray(num_experts, dtype=np.float64)
    np_raw = np.asarray(total_parameter_count, dtype=np.float64)

    # Tiny epsilon to avoid log(0)
    eps = 1e-8

    # Scale parameter count to millions
    ne = ne + eps
    np_mil = np_raw / 1e6 + eps

    # Log-space features
    x = np.log(ne)
    y = np.log(np_mil)

    # Log-quadratic polynomial
    z = (a
         + b * x
         + c * y
         + d * x**2
         + e * y**2
         + f * x * y)

    # Return in original loss scale
    return np.exp(z)


def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log-quadratic scaling law via ordinary least squares.

    Args:
        num_experts: array-like of shape (N,)
        total_parameter_count: array-like of shape (N,)
        loss_values: array-like of shape (N,), observed validation losses

    Returns:
        params_opt: ndarray of length 6 with fitted parameters [a, b, c, d, e, f]
    """
    # Prepare data
    ne = np.asarray(num_experts, dtype=np.float64)
    np_raw = np.asarray(total_parameter_count, dtype=np.float64)
    loss = np.asarray(loss_values, dtype=np.float64)

    # Epsilon to guard logs
    eps = 1e-8

    # Feature construction in log-space
    x = np.log(ne + eps)
    y = np.log(np_raw / 1e6 + eps)

    # Target in log-space
    L = np.log(loss + eps)

    # Design matrix: columns [1, x, y, x^2, y^2, x*y]
    N = x.shape[0]
    X = np.vstack([
        np.ones(N),
        x,
        y,
        x**2,
        y**2,
        x * y
    ]).T  # shape (N, 6)

    # Solve normal equations: params = argmin ||X p - L||^2
    # Use np.linalg.lstsq for stability
    params_opt, *_ = np.linalg.lstsq(X, L, rcond=None)

    return params_opt


# Declare how many parameters the scaling law expects
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END