# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A six-parameter MoE scaling law:
      loss ≈ C + A / [ (num_experts + E)^alpha * ( (total_parameters/1e9) + F )^beta ]
    params = [A, alpha, beta, E, F, C]
    """
    A, alpha, beta, E, F, C = params
    # scale total_parameter_count to billions for numerical stability
    tp = total_parameter_count / 1e9
    # ensure positive denominators
    denom = np.power(num_experts + E, alpha) * np.power(tp + F, beta)
    return C + A / (denom + 1e-12)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the MoE scaling law to data by minimizing MSE with L-BFGS-B and positivity bounds.
    Returns optimized params = [A, alpha, beta, E, F, C].
    """
    # initial guess
    init = np.array([1.0, 0.5, 0.5, 1.0, 1.0, 0.1])
    # bounds: all params >= 0
    bnds = [(1e-8, None),   # A > 0
            (0.0,    None), # alpha >= 0
            (0.0,    None), # beta >= 0
            (0.0,    None), # E >= 0
            (0.0,    None), # F >= 0
            (0.0,    None)] # C >= 0

    def obj(p):
        pred = scaling_law_func(num_experts, total_parameter_count, p)
        return np.mean((pred - loss_values) ** 2)

    res = minimize(obj, init, method='L-BFGS-B', bounds=bnds)
    return res.x if res.success else init

# declare expected parameter count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END