# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    6-parameter MoE scaling law:
      loss ≈ p0 + p1 / [ (n + p2)^p3 * (P_g + p4)^p5 ]
    where:
      n = num_experts (>=1)
      P_g = total_parameter_count / 1e9 (in billions, >=1e-9)
    
    params = [p0, p1, p2, p3, p4, p5]
      p0: asymptotic minimum loss
      p1: scaling amplitude
      p2,p4: horizontal shifts for n and P_g
      p3,p5: exponents on n and P_g
    """
    # ensure positivity/stability
    n = np.maximum(np.asarray(num_experts, dtype=float), 1.0)
    P = np.maximum(np.asarray(total_parameter_count, dtype=float), 1.0)
    P_g = P / 1e9

    p0, p1, p2, p3, p4, p5 = params
    denom = (n + p2) ** p3 * (P_g + p4) ** p5
    # avoid division by zero
    denom = np.maximum(denom, 1e-12)
    return p0 + p1 / denom

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law by nonlinear least squares
    minimizing residuals (predicted - observed).
    """
    n = np.maximum(np.asarray(num_experts, dtype=float), 1.0)
    P = np.maximum(np.asarray(total_parameter_count, dtype=float), 1.0)
    y = np.asarray(loss_values, dtype=float)

    # initial guess
    # p0 ~ 0.9 * min(loss), p1 ~ range(loss) * (mean(n)*mean(P_g))
    P_g = P / 1e9
    y_min, y_max = np.min(y), np.max(y)
    p0_init = max(0.9 * y_min, 1e-6)
    amplitude = max(y_max - p0_init, 1e-3)
    p1_init = amplitude * (np.mean(n) * np.mean(P_g))**0.5
    init = np.array([p0_init, p1_init, 1.0, 0.5, 1.0, 0.5])

    # bounds: all parameters >= 0
    lower = np.zeros(6)
    upper = np.full(6, np.inf)

    def residuals(params):
        pred = scaling_law_func(n, P, params)
        return pred - y

    result = least_squares(
        residuals,
        init,
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000
    )

    if result.success:
        return result.x
    else:
        # fallback to initial guess
        return init

# record parameter count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END