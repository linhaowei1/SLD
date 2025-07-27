# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models
Evolved to use a log–quadratic model in log-space with an analytic fit.
This form uses 6 parameters and captures main effects and interaction
between number of experts and total parameter count.
"""
import numpy as np

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict validation loss at a fixed training step for MoE models,
    using a quadratic model in log-space.

    Model:
        log(L) = a 
               + b * log(E) 
               + c * log(P) 
               + d * log(E) * log(P)
               + e * [log(E)]^2 
               + f * [log(P)]^2
        L = exp(log(L))

    Args:
        num_experts: Array-like of number of experts (E)
        total_parameter_count: Array-like of total parameter counts (P)
        params:    Sequence of 6 parameters [a, b, c, d, e, f]

    Returns:
        Numpy array of predicted loss values.
    """
    E = np.array(num_experts, dtype=np.float64)
    P = np.array(total_parameter_count, dtype=np.float64)

    # Avoid log(0) or negative
    E = np.clip(E, 1.0, None)
    P = np.clip(P, 1.0, None)

    logsE = np.log(E)
    logsP = np.log(P)
    a, b, c, d, e, f = params

    # Compute log-loss and exponentiate
    log_loss = (
        a
        + b * logsE
        + c * logsP
        + d * (logsE * logsP)
        + e * (logsE ** 2)
        + f * (logsP ** 2)
    )
    return np.exp(log_loss)


def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log–quadratic scaling law by ordinary least squares
    in log-space with a tiny ridge term for stability.

    Args:
        num_experts:           Array-like of number of experts
        total_parameter_count: Array-like of total parameter counts
        loss_values:           Array-like of observed loss values

    Returns:
        Numpy array of fitted parameters [a, b, c, d, e, f].
    """
    E = np.array(num_experts, dtype=np.float64)
    P = np.array(total_parameter_count, dtype=np.float64)
    L = np.array(loss_values, dtype=np.float64)

    # Safeguard against non-positive values
    E = np.clip(E, 1.0, None)
    P = np.clip(P, 1.0, None)
    L = np.clip(L, 1e-8, None)

    logsE = np.log(E)
    logsP = np.log(P)
    logL  = np.log(L)

    # Build design matrix X of shape (N, 6)
    # Columns: [1, logE, logP, logE*logP, (logE)^2, (logP)^2]
    X = np.vstack([
        np.ones_like(logL),
        logsE,
        logsP,
        logsE * logsP,
        logsE ** 2,
        logsP ** 2
    ]).T

    # Solve (X^T X + λI) θ = X^T y
    ridge = 1e-6
    XT_X = X.T @ X + ridge * np.eye(6)
    XT_y = X.T @ logL

    params = np.linalg.solve(XT_X, XT_y)
    return params


# Specify number of parameters for external checks
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END