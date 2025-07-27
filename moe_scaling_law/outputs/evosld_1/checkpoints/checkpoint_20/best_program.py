# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models

We fit a 6-parameter log–quadratic scaling law in log-space:
    log L = p0
          + p1 * ln(N)
          + p2 * (ln(N))^2
          + p3 * ln(E)
          + p4 * (ln(E))^2
          + p5 * ln(N) * ln(E)

where:
  - L is validation loss
  - N is total parameter count (dense, expert params excluded)
  - E is number of experts

This form captures main, quadratic, and interaction effects of model size
and expert count. We fit parameters via closed-form least squares in log-space
and optionally refine with a light numerical optimization.
"""

import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict validation loss from number of experts and total parameters.

    Args:
        num_experts: array-like of expert counts (E)
        total_parameter_count: array-like of parameter counts (N)
        params: length-6 array of scaling law coefficients [p0..p5]

    Returns:
        Array of predicted loss values (same shape as inputs).
    """
    # Convert to floats and avoid log(0)
    E = np.array(num_experts, dtype=np.float64) + 1.0
    N = np.array(total_parameter_count, dtype=np.float64) + 1.0

    # Compute logs
    lnE = np.log(E)
    lnN = np.log(N)

    # Unpack parameters
    p0, p1, p2, p3, p4, p5 = params

    # Log–quadratic model
    lnL = p0 \
        + p1 * lnN \
        + p2 * (lnN ** 2) \
        + p3 * lnE \
        + p4 * (lnE ** 2) \
        + p5 * lnN * lnE

    # Return loss in original scale
    return np.exp(lnL)


def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log–quadratic MoE scaling law to data.

    First solves linear least squares in log-space, then optionally
    refines parameters by minimizing MSE in the original space.

    Args:
        num_experts: array-like of expert counts
        total_parameter_count: array-like of parameter counts
        loss_values: array-like of observed loss values

    Returns:
        params: length-6 optimized parameter array [p0..p5]
    """
    # Prepare arrays
    E = np.array(num_experts, dtype=np.float64) + 1.0
    N = np.array(total_parameter_count, dtype=np.float64) + 1.0
    L = np.array(loss_values, dtype=np.float64)

    # Take logs for linear regression
    lnE = np.log(E)
    lnN = np.log(N)
    lnL = np.log(L + 1e-12)

    # Design matrix for [1, lnN, (lnN)^2, lnE, (lnE)^2, lnN*lnE]
    X = np.vstack([
        np.ones_like(lnN),
        lnN,
        lnN ** 2,
        lnE,
        lnE ** 2,
        lnN * lnE
    ]).T

    # Closed-form least squares solution
    params_init, *_ = np.linalg.lstsq(X, lnL, rcond=None)

    # Optional refinement: minimize MSE in original loss space
    def objective(p):
        pred = scaling_law_func(num_experts, total_parameter_count, p)
        return np.mean((pred - L) ** 2)

    result = minimize(objective, params_init, method='L-BFGS-B')
    if result.success:
        return result.x
    else:
        return params_init


# Expose expected parameter count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END