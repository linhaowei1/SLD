# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Small constant to avoid log(0)
_EPS = 1e-8

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict MoE validation loss using a 6-parameter log–quadratic model:
        logL = θ0
             + θ1·log(E)
             + θ2·log(P)
             + θ3·log(E)·log(P)
             + θ4·[log(E)]²
             + θ5·[log(P)]²
        L = exp(logL)
    Inputs:
      num_experts           array-like of expert counts (E)
      total_parameter_count array-like of total params (P)
      params                length-6 array of θ0…θ5
    Returns:
      array of predicted losses
    """
    E = np.asarray(num_experts, dtype=np.float64)
    P = np.asarray(total_parameter_count, dtype=np.float64)
    logE = np.log(np.maximum(E, _EPS))
    logP = np.log(np.maximum(P, _EPS))
    θ0, θ1, θ2, θ3, θ4, θ5 = params
    logL = (
        θ0
        + θ1 * logE
        + θ2 * logP
        + θ3 * (logE * logP)
        + θ4 * (logE ** 2)
        + θ5 * (logP ** 2)
    )
    return np.exp(logL)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log–quadratic scaling law by:
      1) closed-form ridge regression in log-space for initialization
      2) L-BFGS-B refinement to minimize mean squared error in log-space
    Returns the optimized [θ0…θ5].
    """
    # prepare data
    E = np.asarray(num_experts, dtype=np.float64)
    P = np.asarray(total_parameter_count, dtype=np.float64)
    L = np.asarray(loss_values, dtype=np.float64)
    logE = np.log(np.maximum(E, _EPS))
    logP = np.log(np.maximum(P, _EPS))
    logL = np.log(np.maximum(L, _EPS))

    # design matrix: [1, logE, logP, logE*logP, logE^2, logP^2]
    X = np.stack([
        np.ones_like(logE),
        logE,
        logP,
        logE * logP,
        logE ** 2,
        logP ** 2,
    ], axis=1)

    # closed-form ridge init
    XT_X = X.T @ X
    XT_y = X.T @ logL
    ridge_factor = 1e-6 * np.trace(XT_X)
    θ_init = np.linalg.solve(XT_X + ridge_factor * np.eye(6), XT_y)

    # objective: MSE in log-space
    def _obj(theta):
        pred = X.dot(theta)
        return np.mean((pred - logL) ** 2)

    # refine with L-BFGS-B
    res = minimize(
        _obj,
        θ_init,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    θ_opt = res.x if res.success else θ_init
    return θ_opt

# expose parameter count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END