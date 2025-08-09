# EVOLVE-BLOCK-START
"""
Six-parameter log-quadratic scaling law with one final non‐linear refinement.

Model:
    L(E,P) = exp(θ0
                + θ1 * ln P
                + θ2 * ln E
                + θ3 * (ln P * ln E)
                + θ4 * (ln P)**2
                + θ5 * (ln E)**2)

Fitting proceeds in two stages:
  1) closed‐form linear least‐squares on ln(loss)
  2) Levenberg–Marquardt refinement on LHS = loss

This typically improves NMSE/NMAE/R² with negligible extra cost.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Compute the 6-parameter log–quadratic scaling law.
    Inputs:
      data_points: (N,2) array: [num_experts, dense_parameter_count]
      params: length-6 array [θ0…θ5]
    Returns:
      losses: (N,) array of predicted validation loss
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    E = np.clip(X[:,0], 1e-8, None)
    P = np.clip(X[:,1], 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)
    θ = np.asarray(params, dtype=float).ravel()
    if θ.size != 6:
        raise ValueError(f"Expected 6 parameters, got {θ.size}")
    θ0, θ1, θ2, θ3, θ4, θ5 = θ
    u = (θ0
         + θ1 * lnP
         + θ2 * lnE
         + θ3 * (lnP * lnE)
         + θ4 * (lnP ** 2)
         + θ5 * (lnE ** 2))
    return np.exp(u)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter log–quadratic model to (num_experts, dense_params) → loss.
    1) Solve linear least squares in log space.
    2) Refine with a short Levenberg–Marquardt run on the true residuals.
    Returns:
      θ_opt: length-6 array of fitted parameters.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()
    # ensure positive for logs
    y_pos = np.clip(y, 1e-8, None)
    ln_y = np.log(y_pos)
    E = np.clip(X[:,0], 1e-8, None)
    P = np.clip(X[:,1], 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)
    # design matrix for [1, lnP, lnE, lnP*lnE, (lnP)^2, (lnE)^2]
    M = np.vstack([
        np.ones_like(lnP),
        lnP,
        lnE,
        lnP * lnE,
        lnP**2,
        lnE**2
    ]).T  # shape (N,6)
    # closed-form least squares in log space
    θ_init, *_ = np.linalg.lstsq(M, ln_y, rcond=None)
    θ_init = θ_init.ravel()
    # residuals in original-loss space
    def resid(theta):
        return scaling_law_func(X, theta) - y
    # small LM refinement
    try:
        sol = least_squares(resid,
                            θ_init,
                            method='lm',
                            max_nfev=500,
                            ftol=1e-8,
                            xtol=1e-8)
        θ_opt = sol.x
    except Exception:
        # fallback to linear-solution if LM fails
        θ_opt = θ_init
    return θ_opt
# EVOLVE-BLOCK-END