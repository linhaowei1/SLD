import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A 4-parameter saturating power-law (Hill equation) scaling function:
        loss(x) = d + A / (1 + (x / c) ** b)

    where:
        A = (p0)^2          controls the drop from initial loss to floor
        b = (p1)^2          governs the steepness of decay
        c = (p2)^2 + eps    characteristic data scale (eps to avoid zero)
        d = (p3)^2          asymptotic minimum loss

    Args:
        data_points: array-like of training set sizes
        params     : length-4 array [p0, p1, p2, p3]

    Returns:
        numpy array of predicted losses
    """
    x = np.asarray(data_points, dtype=float)
    p0, p1, p2, p3 = params

    # enforce positivity via squaring
    A = p0**2
    b = p1**2
    c = p2**2 + 1e-8
    d = p3**2

    return d + A / (1 + (x / c) ** b)


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