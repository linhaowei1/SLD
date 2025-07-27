# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(proportions, params):
    """
    Predict per‐domain losses from 5‐domain mixture proportions.
    We use 7 parameters per domain:
      Li = A_i
         + B_i * pi^C_i
         + D_i * (1−pi)^E_i
         + F_i * (pi*(1−pi))
         + G_i * log(pi + eps)

    Args:
      proportions: array (n_samples, 5), each row sums to 1
      params:      flat array (35,) = 5 domains × 7 params

    Returns:
      losses: array (n_samples, 5)
    """
    P = np.atleast_2d(proportions)
    if P.shape[1] != 5:
        raise ValueError("Expected proportions shape (n_samples, 5).")
    # reshape into (5 domains, 7 params)
    p = params.reshape(5, 7)
    eps = 1e-8

    # clamp to avoid zero/negatives
    pi = np.clip(P,     eps, 1.0)      # (n,5)
    comp = np.clip(1-P, eps, 1.0)      # (n,5)

    # unpack parameters
    A = p[:, 0]   # (5,)
    B = p[:, 1]
    C = p[:, 2]
    D = p[:, 3]
    E = p[:, 4]
    F = p[:, 5]
    G = p[:, 6]

    # vectorized terms
    term1 = B[None, :] * (pi ** C[None, :])
    term2 = D[None, :] * (comp ** E[None, :])
    term3 = F[None, :] * (pi * comp)
    term4 = G[None, :] * np.log(pi)

    # sum with intercept A
    losses = A[None, :] + term1 + term2 + term3 + term4
    return losses


def fit_scaling_law(proportions, loss_values):
    """
    Fit the 35 parameters of the scaling law via nonlinear least squares.

    Args:
      proportions: array (n_samples, 5)
      loss_values: array (n_samples, 5)

    Returns:
      best_params: flat array (35,)
    """
    P = np.atleast_2d(proportions)
    L = np.atleast_2d(loss_values)
    if P.shape != L.shape or P.shape[1] != 5:
        raise ValueError("Inputs must be shape (n_samples, 5), and match each other.")

    # initial guess: A_i = mean(L_i), B_i..E_i=1, F_i=0, G_i=0
    means = L.mean(axis=0)  # (5,)
    init = np.zeros(5*7)
    for i in range(5):
        init[i*7:(i+1)*7] = [means[i], 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    # per‐parameter bounds
    lower = np.tile([-np.inf,  0.0,   0.0,   0.0,   0.0,  -np.inf, -np.inf], 5)
    upper = np.tile([ np.inf,  np.inf, np.inf, np.inf, np.inf,  np.inf,  np.inf], 5)

    # residuals flattened
    def residuals(x):
        return (scaling_law_func(P, x) - L).ravel()

    best_cost   = np.inf
    best_params = init.copy()

    # multi‐start least squares
    for trial in range(4):
        x0 = init if trial == 0 else init + np.random.randn(35)*0.1
        res = least_squares(residuals, x0,
                            bounds=(lower, upper),
                            xtol=1e-8,
                            ftol=1e-8,
                            verbose=0)
        if res.cost < best_cost:
            best_cost   = res.cost
            best_params = res.x

    return best_params

# attach expected parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END