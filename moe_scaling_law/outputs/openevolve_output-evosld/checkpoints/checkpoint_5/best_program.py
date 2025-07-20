# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Predict MoE validation loss from number of experts and total parameters
    using a compact 6-parameter scaling law:
      loss = c0
           + c1 * P^(−a)
           + c2 * E^(−b)
           + c3 * P^(−a) * E^(−b)
    where:
      P = total_parameter_count (clipped >0 for stability)
      E = num_experts (clipped >0 for stability)
      params = [c0, c1, a, c2, b, c3]
    """
    c0, c1, a, c2, b, c3 = params
    # ensure positive bases
    P = np.maximum(total_parameter_count, 1e-8)
    E = np.maximum(num_experts, 1e-8)
    # compute terms
    t1 = c1 * P**(-a)
    t2 = c2 * E**(-b)
    t3 = c3 * (P**(-a) * E**(-b))
    return c0 + t1 + t2 + t3

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to data by minimizing squared residuals.
    Returns optimized params = [c0, c1, a, c2, b, c3].
    """
    # convert to arrays
    E = np.asarray(num_experts, dtype=float)
    P = np.asarray(total_parameter_count, dtype=float)
    L = np.asarray(loss_values, dtype=float)

    # initial parameter guess
    c0_0 = np.min(L) * 0.5
    spread = np.max(L) - c0_0
    c1_0 = spread * 0.4
    c2_0 = spread * 0.4
    c3_0 = spread * 0.1
    a_0 = 0.3
    b_0 = 0.3
    x0 = np.array([c0_0, c1_0, a_0, c2_0, b_0, c3_0])

    # enforce non-negativity on all params
    lb = np.zeros_like(x0)
    ub = np.full_like(x0, np.inf)

    # residuals: predicted - actual
    def residuals(params):
        return scaling_law_func(E, P, params) - L

    # solve using trust-region reflective algorithm
    res = least_squares(residuals, x0, bounds=(lb, ub), method='trf', max_nfev=5000)

    return res.x if res.success else x0

# indicate expected parameter count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END