# EVOLVE-BLOCK-START
"""
Improved scaling law discovery using a log–quadratic interaction model.
Model: L = exp(θ0 + θ1·lnP + θ2·lnE + θ3·(lnP·lnE) + θ4·(lnP)^2 + θ5·(lnE)^2)
where P = dense parameter count, E = num_experts.
This 6-parameter form captures multiplicative interactions and curvature in log-space.
Fitting is done via least squares on log(loss), avoiding expensive iterative minimization.
"""
import numpy as np

def scaling_law_func(data_points, params):
    # data_points: (N,2) array [num_experts, dense_param_count]
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    # split features
    E = X[:, 0]
    P = X[:, 1]
    # numerical stability: clip to positive
    E = np.clip(E, 1e-8, None)
    P = np.clip(P, 1e-8, None)
    # log features
    lnE = np.log(E)
    lnP = np.log(P)
    # unpack up to 6 parameters
    θ = np.asarray(params, dtype=float).ravel()
    if θ.size != 6:
        raise ValueError(f"Expected 6 parameters, got {θ.size}")
    θ0, θ1, θ2, θ3, θ4, θ5 = θ
    # log‐quadratic model with interaction
    u = θ0 \
        + θ1 * lnP \
        + θ2 * lnE \
        + θ3 * (lnP * lnE) \
        + θ4 * (lnP ** 2) \
        + θ5 * (lnE ** 2)
    # back to original scale
    return np.exp(u)

def fit_scaling_law(data_points, loss_values):
    # Prepare data
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()
    # Ensure strictly positive target for log
    y_pos = np.clip(y, 1e-8, None)
    ln_y = np.log(y_pos)
    # Features
    E = X[:, 0]
    P = X[:, 1]
    E = np.clip(E, 1e-8, None)
    P = np.clip(P, 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)
    # Design matrix: [1, lnP, lnE, lnP·lnE, (lnP)^2, (lnE)^2]
    M = np.vstack([
        np.ones_like(lnP),
        lnP,
        lnE,
        lnP * lnE,
        lnP**2,
        lnE**2
    ]).T  # shape (N,6)
    # Solve linear least squares: minimize ||M·θ - ln_y||^2
    θ_opt, *_ = np.linalg.lstsq(M, ln_y, rcond=None)
    # Return fitted parameters θ0…θ5
    return θ_opt
# EVOLVE-BLOCK-END