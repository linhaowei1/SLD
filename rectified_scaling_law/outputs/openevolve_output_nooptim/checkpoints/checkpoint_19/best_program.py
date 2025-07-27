import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A 4-parameter “shifted & scaled” power law with an asymptotic floor:
        loss(x) = d + a * (1 + x/c)^(-b)

    where:
        a > 0  : controls the overall decay amplitude
        b > 0  : governs the power-law decay rate
        c > 0  : characteristic scale for x
        d ≥ 0  : asymptotic minimum loss

    Args:
        data_points: array-like of training set sizes
        params     : length-4 array [p0, p1, p2, p3]

    Returns:
        numpy array of predicted losses
    """
    x = np.asarray(data_points, dtype=float)
    p0, p1, p2, p3 = params

    # enforce positivity via squaring
    a = p0**2
    b = p1**2
    c = p2**2 + 1e-8   # tiny epsilon to prevent division by zero
    d = p3**2

    return d + a * np.power(1 + x / c, -b)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to observed (x, loss) pairs.
    Uses BFGS starting from ones() for all params.
    """
    initial_params = np.ones(4)

    def objective(params):
        try:
            pred = scaling_law_func(data_points, params)
            return np.mean((pred - loss_values) ** 2)
        except Exception:
            return 1e6  # large penalty if numerical issues arise

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# Tell downstream code we expect 4 parameters
scaling_law_func.num_params = 4