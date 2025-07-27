# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A 6-parameter scaling law for MoE models capturing joint
    effects of experts and total parameter count.

    loss = a0 + (a1 * E^c1 + a2 * P^c2)^(-c3)

    where:
      E = num_experts,
      P = total_parameter_count,
      params = [a0, a1, c1, a2, c2, c3].
    """
    a0, a1, c1, a2, c2, c3 = params
    # avoid zero or negative inputs
    E = np.maximum(num_experts, 1e-6)
    P = np.maximum(total_parameter_count, 1e-6)
    # combined capacity term
    cap = a1 * (E ** c1) + a2 * (P ** c2)
    # numerical floor for stability
    cap = np.maximum(cap, 1e-12)
    # final loss prediction
    loss_pred = a0 + cap ** (-c3)
    return loss_pred

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to data by minimizing MSE.
    Returns optimized [a0, a1, c1, a2, c2, c3].
    """
    # convert to numpy arrays
    E = np.array(num_experts, dtype=np.float64)
    P = np.array(total_parameter_count, dtype=np.float64)
    L = np.array(loss_values, dtype=np.float64)
    # initialize parameters
    init_a0 = max(np.min(L) * 0.9, 1e-3)
    init_params = np.array([
        init_a0,     # a0 >= 0
        1e-3,        # a1 > 0
        0.5,         # c1 > 0
        1e-9,        # a2 > 0
        0.33,        # c2 > 0
        0.5          # c3 > 0
    ], dtype=np.float64)
    # bounds to enforce positivity and reasonable exponents
    bounds = [
        (0.0, None),      # a0
        (1e-12, None),    # a1
        (1e-8, 10.0),     # c1
        (1e-12, None),    # a2
        (1e-8, 10.0),     # c2
        (1e-8, 10.0),     # c3
    ]
    # objective: mean squared error
    def objective(p):
        pred = scaling_law_func(E, P, p)
        return np.mean((pred - L) ** 2)

    res = minimize(
        objective,
        init_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12}
    )

    if res.success:
        return res.x
    else:
        # fallback to initial if optimization fails
        return init_params

# specify number of parameters used
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END