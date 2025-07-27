# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    6-parameter log-quadratic MoE scaling law:
      let x = log(num_experts), y = log(total_parameters)
      z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
      loss = exp(z)
    params = [a, b, c, d, e, f]
    """
    eps = 1e-12
    x = np.log(num_experts + eps)
    y = np.log(total_parameter_count + eps)
    a, b, c, d, e, f = params
    z = a + b*x + c*y + d*(x**2) + e*(y**2) + f*(x*y)
    return np.exp(z)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log-quadratic form by linear least squares in log space.
    Solves for params in:
      log(loss) ≈ a + b*log(ne) + c*log(N) + d*log(ne)^2 + e*log(N)^2 + f*log(ne)*log(N)
    """
    eps = 1e-12
    x = np.log(num_experts + eps)
    y = np.log(total_parameter_count + eps)
    z = np.log(loss_values + eps)
    # build design matrix [1, x, y, x^2, y^2, x*y]
    X = np.vstack((np.ones_like(x), x, y, x**2, y**2, x*y)).T
    # solve ordinary least squares
    params, *_ = np.linalg.lstsq(X, z, rcond=None)
    return params

# declare how many params the function uses
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END