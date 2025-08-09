import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    4-parameter shifted power law:
      L(N) = B + A * (N + C)^(-alpha)
    params = [lnA, ln_alpha, lnC, lnB] (log-domain).
    """
    X = np.asarray(data_points).reshape(-1).astype(float)
    p = np.asarray(params).reshape(-1)
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    # Recover positive parameters
    A, alpha, C, B = np.exp(p)
    return B + A * (X + C) ** (-alpha)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law L(N)=B + A*(N+C)^(-alpha) by
    minimizing a weighted sum of absolute MSE and relative (log) MSE
    using analytic gradients for speed and stability.
    Returns optimized log-domain params [lnA, ln_alpha, lnC, lnB].
    """
    X = np.asarray(data_points).reshape(-1).astype(float)
    y = np.asarray(loss_values).reshape(-1).astype(float)
    eps = 1e-12

    # Initial natural-domain guesses
    y_min, y_max = y.min(), y.max()
    B0 = max(0.9 * y_min, eps)
    C0 = max(0.05 * np.median(X), eps)
    # Quick log-log fit for A and alpha
    y_shift = np.clip(y - B0, eps, None)
    logX = np.log(X + C0)
    logY = np.log(y_shift)
    slope, intercept = np.polyfit(logX, logY, 1)
    alpha0 = max(-slope, 1e-3)
    A0 = max(np.exp(intercept), eps)

    # Log-domain starting point
    p0 = np.log([A0, alpha0, C0, B0])

    # Weights to emphasize larger data sizes
    w = X / X.sum()
    # Balance between absolute and relative error
    lam = 0.5

    def obj_and_grad(p):
        # Unpack positive parameters
        A, alpha, C, B = np.exp(p)
        Xc = X + C
        pred = B + A * Xc**(-alpha) + eps

        # Residuals
        r = pred - y
        lr = np.log(pred) - np.log(y + eps)

        # Objective: weighted absolute MSE + lam * weighted log-MSE
        obj = (1 - lam) * np.sum(w * r**2) + lam * np.sum(w * lr**2)

        # Derivatives w.r.t. log-parameters
        # ∂pred/∂logA = A * Xc^(-alpha)
        d_logA = A * Xc**(-alpha)
        # ∂pred/∂log_alpha = (∂pred/∂alpha)*alpha = -A*alpha*Xc^(-alpha)*ln(Xc)
        d_logα = -A * alpha * Xc**(-alpha) * np.log(Xc)
        # ∂pred/∂logC = (∂pred/∂C)*C = -A*alpha*C*Xc^(-alpha-1)
        d_logC = -A * alpha * C * Xc**(-alpha - 1)
        # ∂pred/∂logB = B
        d_logB = B * np.ones_like(X)

        # Gradient components
        # from absolute MSE
        grad_abs = 2 * (1 - lam) * np.array([
            np.sum(w * r * d_logA),
            np.sum(w * r * d_logα),
            np.sum(w * r * d_logC),
            np.sum(w * r * d_logB),
        ])
        # from log MSE
        grad_rel = 2 * lam * np.array([
            np.sum(w * lr * (d_logA / pred)),
            np.sum(w * lr * (d_logα / pred)),
            np.sum(w * lr * (d_logC / pred)),
            np.sum(w * lr * (d_logB / pred)),
        ])

        return obj, grad_abs + grad_rel

    # Optimize with L-BFGS-B and analytic gradient
    res = minimize(fun=lambda p: obj_and_grad(p)[0],
                   x0=p0,
                   jac=lambda p: obj_and_grad(p)[1],
                   method="L-BFGS-B")

    return res.x if res.success else p0
# EVOLVE-BLOCK-END