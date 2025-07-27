# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models.
Evolved version with a 6-parameter form capturing expert, parameter
and cross-term effects, fitted via bounded L-BFGS-B optimization.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict MoE validation loss given experts and parameter count.

    Functional form:
        loss = a
             + b * (num_experts + eps)^(-alpha)
             + c * (total_parameter_count + eps)^(-beta)
             + d * (num_experts + eps)^(-alpha) * (total_parameter_count + eps)^(-beta)

    Uses 6 parameters: [a, b, c, d, alpha, beta].

    Args:
        num_experts: array-like of expert counts
        total_parameter_count: array-like of dense parameter counts
        params: array-like of length 6

    Returns:
        Array of predicted losses.
    """
    a, b, c, d, alpha, beta = params
    eps = 1e-8
    e_term = np.power(num_experts + eps, -alpha)
    p_term = np.power(total_parameter_count + eps, -beta)
    return a + b * e_term + c * p_term + d * (e_term * p_term)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to empirical data.

    Minimizes MSE between predicted and actual loss using L-BFGS-B
    with bounds to ensure stability and interpretability.

    Args:
        num_experts: array-like of expert counts
        total_parameter_count: array-like of dense parameter counts
        loss_values: array-like of observed losses

    Returns:
        Optimized parameter vector of length 6.
    """
    ne = np.asarray(num_experts, dtype=np.float64)
    tp = np.asarray(total_parameter_count, dtype=np.float64)
    lv = np.asarray(loss_values, dtype=np.float64)

    # Initial guess: a ~ min observed loss, others positive
    a0 = np.min(lv) * 0.9
    b0, c0, d0 = (np.ptp(lv),) * 3  # dynamic range
    alpha0, beta0 = 0.5, 0.5
    x0 = np.array([a0, b0, c0, d0, alpha0, beta0], dtype=np.float64)

    # Bounds: a free, b,c,d >=0, alpha,beta in [1e-3, 5]
    bounds = [
        (None, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (1e-3, 5.0),
        (1e-3, 5.0),
    ]

    def mse_obj(x):
        pred = scaling_law_func(ne, tp, x)
        return np.mean((pred - lv) ** 2)

    res = minimize(
        mse_obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 10000},
    )

    if res.success:
        return res.x
    else:
        return x0

# Number of parameters in the scaling law
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END