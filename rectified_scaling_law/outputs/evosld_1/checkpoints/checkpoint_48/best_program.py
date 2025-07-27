import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    4-parameter normalized rational power‐law scaling:
        L(N) = a / ((N/N_max)^b + d) + c

    where:
      - N is the training data size,
      - N_max = max(data_points),
      - a > 0 controls amplitude,
      - b > 0 is the power‐law exponent,
      - d > 0 is a stabilizing offset in the denominator,
      - c ≥ 0 is the asymptotic floor.
    """
    N = np.asarray(data_points, dtype=np.float64)
    a, b, c, d = params

    # enforce parameter constraints
    eps = 1e-12
    a = max(a, eps)
    b = max(b, eps)
    c = max(c, 0.0)
    d = max(d, eps)

    N_max = np.max(N)
    # normalize to [0,1]
    x = N / (N_max + eps)

    return a / (np.power(x, b) + d) + c


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law:
       L(N) = a / ((N/N_max)^b + d) + c
    to (data_points, loss_values) by minimizing a robust relative MSE.
    Returns optimized params [a, b, c, d].
    """
    N = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    eps = 1e-12

    # data-derived scales
    N_max = np.max(N)
    y_min, y_max = np.min(y), np.max(y)
    amp = max(y_max - y_min, eps)

    # Parameter bounds
    bounds = [
        (eps,       amp * 5.0),  # a: amplitude up to 5×range
        (eps,       10.0),       # b: exponent
        (0.0,       y_max),      # c: floor up to max loss
        (eps,       10.0)        # d: denom offset
    ]

    # objective: mean squared relative error
    def objective(p):
        y_pred = scaling_law_func(N, p)
        return np.mean(((y_pred - y) / (y + eps)) ** 2)

    # 1) Global search: differential evolution
    de_res = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        popsize=15,
        maxiter=40,
        tol=1e-5,
        seed=42,
        polish=False
    )

    # 2) Local refinement from DE result
    local_from_de = minimize(
        objective,
        de_res.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-9, 'maxiter':200}
    )

    # 3) Local refinement from a heuristic initial guess
    #    a ~ amp, b~1.0, c~y_min, d~1.0
    x0 = np.array([amp, 1.0, y_min, 1.0], dtype=np.float64)
    local_from_guess = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-9, 'maxiter':200}
    )

    # Collect candidates
    candidates = []
    # global
    candidates.append((de_res.fun, de_res.x))
    # local if successful
    if local_from_de.success:
        candidates.append((local_from_de.fun, local_from_de.x))
    if local_from_guess.success:
        candidates.append((local_from_guess.fun, local_from_guess.x))

    # pick best by objective
    best_fun, best_params = min(candidates, key=lambda tup: tup[0])

    return np.asarray(best_params, dtype=np.float64)


# Expose the number of parameters
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END