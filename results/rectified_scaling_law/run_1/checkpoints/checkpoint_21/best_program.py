import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predicts loss using a 4-parameter shifted power law in log-domain:
      L(N) = B + A * (N + C)^(-alpha)
    where params = [logA, logα, logC, logB].
    """
    # Flatten inputs
    D = np.asarray(data_points).ravel().astype(float)
    p = np.asarray(params).ravel()
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    # Reconstruct strictly positive parameters
    A     = np.exp(p[0])    # amplitude
    alpha = np.exp(p[1])    # decay exponent
    C     = np.exp(p[2])    # horizontal shift
    B     = np.exp(p[3])    # asymptotic floor
    # Compute predictions
    return B + A * (D + C) ** (-alpha)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the 4-parameter scaling law to (data_points, loss_values).
    Returns optimized params in log-domain [logA, logα, logC, logB].
    """
    D = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # Heuristic initial guesses in the natural domain
    y_min, y_max = y.min(), y.max()
    A0     = max(y_max - y_min, 1e-3)
    alpha0 = 0.5
    C0     = max(np.median(D), 1.0)
    B0     = max(y_min, 1e-3)

    # Switch to log-domain to enforce positivity
    p0 = np.log([A0, alpha0, C0, B0])

    # Mean squared error objective in log-domain
    def _objective(p):
        pred = scaling_law_func(D, p)
        return np.mean((pred - y) ** 2)

    # Optimize with L-BFGS-B for robustness
    result = minimize(
        _objective,
        p0,
        method='L-BFGS-B',
        options={
            'ftol': 1e-10,
            'gtol': 1e-08,
            'maxiter': 5000
        }
    )

    # Return best log-domain parameters (or initial if convergence failed)
    return result.x if result.success else p0
# EVOLVE-BLOCK-END