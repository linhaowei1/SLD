# EVOLVE-BLOCK-START
"""
Simplified, robust scaling‐law via polynomial regression in log‐space.

We model Lossu as a quadratic function of log‐normalized
vocabulary size, non‐vocab parameters, and character count:

  Lossu ≈ p0
        + p1·ℓV + p2·ℓP + p3·ℓC
        + p4·ℓV² + p5·ℓP² + p6·ℓC²

where ℓV = log(vocab_size / max_vocab), etc.
This 7‐parameter form captures power‐law trends and curvature,
fits via a single linear least‐squares solve, and avoids iterative
optimizers—improving readability, stability, and speed.
"""
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu from model/training scale via log‐space quadratic form.

    Inputs:
      Non_vocab_parameters: array of non‐vocab parameter counts
      vocab_size         : array of vocabulary sizes
      num_characters     : array of characters processed
      params             : array of 7 coefficients [p0…p6]

    Returns:
      Predicted Lossu array.
    """
    # normalize each input by its maximum
    Vn = vocab_size / np.max(vocab_size)
    Pn = Non_vocab_parameters / np.max(Non_vocab_parameters)
    Cn = num_characters / np.max(num_characters)
    # take logs
    lv = np.log(Vn)
    lp = np.log(Pn)
    lc = np.log(Cn)
    # design matrix: [1, ℓV, ℓP, ℓC, ℓV², ℓP², ℓC²]
    X = np.column_stack((np.ones_like(lv), lv, lp, lc, lv**2, lp**2, lc**2))
    return X.dot(params)

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7‐parameter quadratic log‐space scaling law by least squares.

    Returns:
      params: array of 7 optimized coefficients [p0…p6].
    """
    # normalize and log‐transform inputs
    Vn = vocab_size / np.max(vocab_size)
    Pn = Non_vocab_parameters / np.max(Non_vocab_parameters)
    Cn = num_characters / np.max(num_characters)
    lv = np.log(Vn)
    lp = np.log(Pn)
    lc = np.log(Cn)
    # build the regression matrix
    X = np.column_stack((np.ones_like(lv), lv, lp, lc, lv**2, lp**2, lc**2))
    # solve least squares in one shot
    params, *_ = np.linalg.lstsq(X, lossu_values, rcond=None)
    return params

# record parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END