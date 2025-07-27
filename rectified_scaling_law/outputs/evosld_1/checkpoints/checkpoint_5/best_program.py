# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law:
        L(N) = a * (N_norm + d)^(-b) + c
    where N_norm = N / max(N)
    params = [a, b, c, d]
      a > 0       (scale)
      b >= 0      (exponent)
      c >= 0      (irreducible loss floor)
      d > 0       (offset for stability)
    """
    a, b, c, d = params
    x = np.asarray(data_points, dtype=float)
    max_x = np.max(x)
    if max_x <= 0:
        max_x = 1.0
    # normalize to [0,1]
    x = x / max_x
    # offset and clip for numerical stability
    x = np.clip(x + d, 1e-8, None)
    return a * x**(-b) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law to (data_points, loss_values)
    using multi-start bounded L-BFGS-B with a small L2 regularizer.
    
    Returns optimized params = [a, b, c, d]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Bounds:     a ∈ [1e-8, ∞), b ∈ [0, 5], c ∈ [0, ∞), d ∈ [1e-8, 1]
    bounds = [
        (1e-8, None),
        (0.0, 5.0),
        (0.0, None),
        (1e-8, 1.0),
    ]

    def objective(params):
        pred = scaling_law_func(x, params)
        mse = np.mean((pred - y)**2)
        # small L2 regularization to stabilize fitting
        reg = 1e-6 * np.sum(params**2)
        return mse + reg

    best_loss = np.inf
    best_params = None
    rng = np.random.default_rng(seed=0)

    # Multi-start optimization
    for _ in range(8):
        init = np.array([
            rng.uniform(low, high if high is not None else low + 1.0)
            for (low, high) in bounds
        ])
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'maxiter': 5000}
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            continue

    # Fallback if all restarts fail
    if best_params is None:
        # a=1, b=0.5, c=min(loss), d=1e-2
        best_params = np.array([1.0, 0.5, np.min(y), 1e-2], dtype=float)

    return best_params

# Declare how many parameters this model uses
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END