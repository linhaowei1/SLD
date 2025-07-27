import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A flexible 4-parameter double power-law scaling law:
        loss(x) = a * x^(-b) + c * x^(-d)

    We enforce positivity of all coefficients and exponents by
    squaring the raw parameters internally.

    Args:
        data_points: array-like of training data sizes
        params: array-like of 4 raw parameters [p0, p1, p2, p3]

    Returns:
        np.ndarray of predicted loss values
    """
    x = np.asarray(data_points, dtype=float)
    # Raw params
    p0, p1, p2, p3 = np.asarray(params, dtype=float)
    # Enforce positivity
    a = p0 ** 2   # amplitude of first power law
    b = p1 ** 2   # exponent of first power law
    c = p2 ** 2   # amplitude of second power law
    d = p3 ** 2   # exponent of second power law
    # Compute double-term power-law
    return a * np.power(x, -b) + c * np.power(x, -d)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law to (data_points, loss_values) pairs via BFGS minimizing MSE.

    Args:
        data_points: array-like of training data sizes
        loss_values: array-like of observed loss values

    Returns:
        np.ndarray of optimized raw parameters [p0, p1, p2, p3]
    """
    # Initialize all 4 raw parameters to 1
    initial_params = np.ones(4, dtype=float)

    def objective(params):
        try:
            pred = scaling_law_func(data_points, params)
            return np.mean((pred - loss_values) ** 2)
        except Exception:
            # Large penalty if numerical issues
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# Declare the number of parameters our scaling law expects
scaling_law_func.num_params = 4