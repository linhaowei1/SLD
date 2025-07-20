# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    6-parameter log-quadratic scaling law for MoE validation loss:
      x = log(E + eps)
      y = log(P + eps)
      log(loss) = θ0
                + θ1·x
                + θ2·y
                + θ3·x·y
                + θ4·x^2
                + θ5·y^2
      loss = exp(log(loss))
    Args:
        num_experts: array-like of expert counts E
        total_parameter_count: array-like of total params P
        params: [θ0, θ1, θ2, θ3, θ4, θ5]
    Returns:
        pred_loss: array of predicted losses
    """
    θ0, θ1, θ2, θ3, θ4, θ5 = params
    eps = 1e-8
    E = np.log(np.asarray(num_experts, dtype=np.float64) + eps)
    P = np.log(np.asarray(total_parameter_count, dtype=np.float64) + eps)
    log_l = (θ0
             + θ1 * E
             + θ2 * P
             + θ3 * E * P
             + θ4 * E**2
             + θ5 * P**2)
    return np.exp(log_l)

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter log-quadratic scaling law by minimizing MSE.
    Uses a linearized least-squares init for θ, then refines with L-BFGS-B.
    Args:
        num_experts: array of expert counts
        total_parameter_count: array of total params
        loss_values: array of observed validation losses
    Returns:
        params_opt: ndarray of length 6
    """
    E_raw = np.asarray(num_experts, dtype=np.float64)
    P_raw = np.asarray(total_parameter_count, dtype=np.float64)
    L_raw = np.asarray(loss_values, dtype=np.float64)
    eps = 1e-8

    # Precompute logs
    X = np.log(E_raw + eps)
    Y = np.log(P_raw + eps)
    Z = np.log(L_raw + eps)

    # Build design matrix for initial linear fit: [1, X, Y, X*Y, X^2, Y^2]
    M = np.vstack([np.ones_like(X), X, Y, X*Y, X**2, Y**2]).T
    # Solve least squares for initial θ
    θ_init, *_ = np.linalg.lstsq(M, Z, rcond=None)
    
    # Objective: MSE between predicted loss and true loss
    def mse_obj(θ):
        pred = scaling_law_func(E_raw, P_raw, θ)
        if not np.all(np.isfinite(pred)):
            return 1e6
        return np.mean((pred - L_raw)**2)
    
    # Bounds to ensure numeric stability
    bnds = [(-10.0, 10.0)] * 6

    res = minimize(
        mse_obj,
        θ_init,
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': 5000, 'ftol': 1e-12}
    )

    if res.success:
        params_opt = res.x
    else:
        params_opt = θ_init

    return params_opt

# Attach metadata
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END