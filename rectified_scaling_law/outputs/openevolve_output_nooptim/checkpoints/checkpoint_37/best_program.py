import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A 4-parameter Hill-type saturating scaling law:
        loss(x) = d + a / (1 + (x / c)**b)

    where:
        a ≥ 0 controls the initial decay amplitude,
        b ≥ 0 is the decay exponent (slope),
        c > 0 is the half-saturation constant,
        d ≥ 0 is the asymptotic minimum loss.

    We map the unconstrained params to positive via squaring:
        a = params[0]**2
        b = params[1]**2
        c = params[2]**2 + ε
        d = params[3]**2
    to ensure the function is well-behaved for all real inputs.
    """
    x = np.asarray(data_points, dtype=float)
    p0, p1, p2, p3 = params

    # enforce positivity
    a = p0 * p0
    b = p1 * p1
    c = p2 * p2 + 1e-12  # avoid division by zero
    d = p3 * p3

    # Hill-type saturating form
    return d + a / (1.0 + np.power(x / c, b))

# Inform the fitter how many parameters this function uses
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