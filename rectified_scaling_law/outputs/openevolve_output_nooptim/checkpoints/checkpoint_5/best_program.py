import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A 4-parameter double power-law scaling law:
        loss(x) = a * x^(-b) + c * x^(-d)

    This form captures two decay regimes—one dominating at smaller
    data sizes and another at larger sizes—allowing more flexible
    fitting of empirical loss curves.

    Args:
        data_points: array-like of training set sizes (all > 0)
        params:      array-like of 4 parameters [a, b, c, d]
                     a, c > 0 scale the two regimes,
                     b, d > 0 are their respective exponents.

    Returns:
        Array of predicted loss values (same shape as data_points).
    """
    x = np.asarray(data_points, dtype=float)
    a, b, c, d = params
    # two power-law terms
    return a * np.power(x, -b) + c * np.power(x, -d)

# Attach the expected parameter count
scaling_law_func.num_params = 4

def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law to observed losses by minimizing MSE via BFGS.

    Args:
        data_points:  array-like of training set sizes
        loss_values:  array-like of observed losses

    Returns:
        Optimized parameter vector of length 4.
    """
    # Initialize all four parameters to 1
    initial_params = np.ones(4)

    def objective(params):
        try:
            pred = scaling_law_func(data_points, params)
            return np.mean((pred - loss_values) ** 2)
        except Exception:
            # Infeasible params get a large penalty
            return 1e6

    # Perform BFGS optimization
    result = minimize(objective, initial_params, method='BFGS')

    if result.success:
        return result.x
    else:
        # Fallback to initial guess if optimization fails
        return initial_params