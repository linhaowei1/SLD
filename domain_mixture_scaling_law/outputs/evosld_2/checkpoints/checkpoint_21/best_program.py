"""
Refined domain‐mixture scaling law for LLM training losses.
Each domain loss Li is modeled as a nonlinear power‐law of the negative‐log proportions:

    Let x_j = -log(p_j + ε).
    For domain i:
      Li = α_i
         + ∑_{j=1..5} β_{ij} * x_j^{δ_i}

Parameters per domain (7):
  α_i       : bias term
  δ_i       : shared exponent
  β_{ij}     : weight on x_j^δ_i  (j=1..5)

Total parameters = 5 domains × 7 = 35.

We fit each domain separately via bounded nonlinear least squares
to capture curvature while keeping parameter count low.
"""
import numpy as np
from scipy.optimize import least_squares

EPS = 1e-8

def scaling_law_func(proportions, params):
    """
    Predict per‐domain losses given mixture proportions and parameters.
    Args:
        proportions: array‐like of shape [n_samples, 5], rows sum to 1.
        params     : 1D array of length 35 (5 domains × 7 params each).
    Returns:
        losses: np.ndarray of shape [n_samples, 5].
    """
    P = np.atleast_2d(proportions).astype(float)
    n_samples, n_dom = P.shape
    if n_dom != 5:
        raise ValueError(f"Expected 5 domains, got {n_dom}")
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 35:
        raise ValueError(f"Expected 35 params, got {p.size}")
    # feature matrix: x = -log(p + ε)
    X = -np.log(P + EPS)  # shape [n_samples, 5]
    losses = np.zeros((n_samples, 5), dtype=float)
    # evaluate each domain
    for i in range(5):
        pi = p[i*7:(i+1)*7]
        alpha = pi[0]
        delta = pi[1]
        beta  = pi[2:]          # shape (5,)
        # x^delta
        Xd = X**delta           # broadcast power
        losses[:, i] = alpha + Xd.dot(beta)
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling‐law parameters via bounded nonlinear least squares.
    Args:
        proportions: array‐like [n_samples, 5]
        loss_values: array‐like [n_samples, 5]
    Returns:
        params: 1D np.ndarray of length 35.
    """
    P = np.atleast_2d(proportions).astype(float)
    L = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = P.shape
    if n_dom != 5 or L.shape != (n_samples, 5):
        raise ValueError("Expect shapes [n_samples,5] for proportions and loss_values")
    # feature matrix
    X = -np.log(P + EPS)  # [n_samples,5]
    all_params = np.zeros((5, 7), dtype=float)
    # fit each domain independently
    for i in range(5):
        yi = L[:, i]
        # initial linear regression to get α and β_j (assume δ=1)
        A = np.column_stack([np.ones(n_samples), X])  # [n_samples,6]
        lin_coef, *_ = np.linalg.lstsq(A, yi, rcond=None)
        alpha0 = lin_coef[0]
        beta0  = lin_coef[1:]
        delta0 = 1.0
        p0 = np.hstack([alpha0, delta0, beta0])  # initial guess

        # residual function for domain i
        def residual(params_i):
            a, d = params_i[0], params_i[1]
            b = params_i[2:]
            # clamp exponent to avoid numeric issues
            dx = X**d
            return a + dx.dot(b) - yi

        # bounds: α free, δ in [0.1, 5], β_j free
        lower = np.hstack([-np.inf, 0.1, np.full(5, -np.inf)])
        upper = np.hstack([ np.inf, 5.0, np.full(5,  np.inf)])
        sol = least_squares(
            residual, p0,
            bounds=(lower, upper),
            xtol=1e-8, ftol=1e-8, gtol=1e-8,
            verbose=0
        )
        all_params[i, :] = sol.x

    return all_params.ravel()

# declare expected param count
scaling_law_func.num_params = 35