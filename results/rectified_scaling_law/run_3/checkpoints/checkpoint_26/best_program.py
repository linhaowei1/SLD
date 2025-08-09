import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predicts loss as a function of data size N using a four-parameter
    rational‐power scaling law:
        L(N) = B + A / (C + N^α)
    All parameters {A, α, C, B} are constrained positive by
    an exp–reparameterization of the input `params`.
    
    Args:
      data_points: array‐like of shape (N,) or (N,1), the data sizes.
      params: array‐like of length 4 (pA, pα, pC, pB) in log‐domain.
      
    Returns:
      losses: np.ndarray of shape (N,), the predicted losses.
    """
    X = np.asarray(data_points).reshape(-1)
    p = np.asarray(params).ravel()
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters, got {p.size}")
    # ensure positivity
    A     = np.exp(p[0])   # amplitude
    α     = np.exp(p[1])   # exponent
    C     = np.exp(p[2])   # offset in denominator
    B     = np.exp(p[3])   # asymptotic floor
    # compute model
    return B + A / (C + np.power(X, α))


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law L(N) = B + A/(C + N^α) to data.
    Uses SciPy's Levenberg–Marquardt (via least_squares) with simple
    positivity bounds on the natural parameters in log‐domain.
    
    Args:
      data_points: array‐like of shape (N,) or (N,1), the data sizes.
      loss_values: array‐like of shape (N,), the observed losses.
      
    Returns:
      p_opt: np.ndarray of length 4, optimized parameters
             in the same log‐domain form accepted by scaling_law_func.
    """
    X = np.asarray(data_points).reshape(-1)
    y = np.asarray(loss_values).reshape(-1)
    
    # Initial natural‐domain guesses
    # B0 ~ smallest observed loss (floor)
    B0 = max(y.min(), 1e-6)
    # C0 small positive constant
    C0 = max(1.0, 1e-6)
    # α0 moderate decay exponent
    α0 = 0.5
    # A0 chosen so that L(X.min) ~ A/(C0 + X.min^α0) + B0 ≈ max(y)
    A0 = max((y.max() - B0) * (C0 + X.min()**α0), 1e-6)
    
    # log‐domain starting point
    p0 = np.log([A0, α0, C0, B0])
    
    # residual function for least_squares
    def _residual(p):
        return scaling_law_func(X, p) - y
    
    # lower/upper bounds in log‐domain to keep natural params > 0
    lower = np.log([1e-6, 1e-3, 1e-6, 1e-6])
    upper = np.log([1e6, 10.0, 1e6, 1e6])
    
    result = least_squares(
        _residual,
        p0,
        bounds=(lower, upper),
        method='trf',
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        verbose=0
    )
    
    # if optimization failed, return initial guess
    return result.x if result.success else p0