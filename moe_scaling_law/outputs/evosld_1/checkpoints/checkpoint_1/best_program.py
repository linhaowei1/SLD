# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    Mixture-of-Experts scaling law with up to 6 parameters:
      params = [a, b, p, c, q, d]
    Loss = a 
           + b / (E ** p) 
           + c / (P ** q) 
           + d / (E ** p * P ** q)
    where E = num_experts + eps, P = total_parameter_count + eps.
    """
    a, b, p, c, q, d = params
    eps = 1e-8
    E = np.maximum(num_experts, 0) + eps
    P = np.maximum(total_parameter_count, 0) + eps
    return a + b / (E**p) + c / (P**q) + d / (E**p * P**q)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law by minimizing MSE.
    We impose non-negativity on b,c,d,p,q via bounds, while a is free.
    """
    # Initial guess: [a, b, p, c, q, d]
    init_a = np.median(loss_values)
    init = np.array([init_a, 1.0, 0.5, 1.0, 0.5, 0.1], dtype=float)
    
    # Bounds: a free, others >= 0
    bounds = [
        (None, None),  # a
        (0.0, None),   # b
        (0.0, None),   # p
        (0.0, None),   # c
        (0.0, None),   # q
        (0.0, None),   # d
    ]
    
    def objective(params):
        pred = scaling_law_func(num_experts, total_parameter_count, params)
        return np.mean((pred - loss_values) ** 2)
    
    result = minimize(
        objective,
        init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-12}
    )
    
    if result.success:
        return result.x
    else:
        # Fallback to initial guess if optimization fails
        return init

# Record number of parameters
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END