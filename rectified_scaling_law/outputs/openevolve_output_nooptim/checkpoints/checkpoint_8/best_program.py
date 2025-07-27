# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM fine‐tuning scenarios

We adopt a 4‐parameter “shifted & scaled” power‐law with an asymptotic floor:
    loss(x) = d + a * (1 + x/c)^(-b)

Here:
  • a = exp(p0)  controls the overall amplitude of the decay
  • b = exp(p1)  is the decay exponent
  • c = exp(p2)  sets the characteristic scale of x
  • d = exp(p3)  is the asymptotic minimum loss (floor)

All four parameters are optimized in unconstrained space and
pushed positive via an exponential transform, ensuring a smooth,
monotonic decay to a positive floor.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    4‐parameter scaled power‐law with offset.

    Args:
        data_points: array‐like of training set sizes (floats or ints)
        params:       length‐4 array [p0, p1, p2, p3] in unconstrained space

    Returns:
        numpy array of predicted losses
    """
    x = np.asarray(data_points, dtype=float)
    p0, p1, p2, p3 = params

    # enforce positivity via exponential transform
    a = np.exp(p0)   # amplitude
    b = np.exp(p1)   # exponent
    c = np.exp(p2)   # x‐scale
    d = np.exp(p3)   # loss floor

    return d + a * np.power(1 + x / c, -b)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4‐parameter scaling law to observed losses.
    Uses BFGS optimization starting from all‐ones initial guess.
    """
    initial_params = np.ones(4)

    def objective(params):
        try:
            pred = scaling_law_func(data_points, params)
            return np.mean((pred - loss_values) ** 2)
        except Exception:
            # large penalty on invalid params
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# declare how many params the scaling law expects
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END