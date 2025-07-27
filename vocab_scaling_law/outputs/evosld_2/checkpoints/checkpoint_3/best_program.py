# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Log‐polynomial scaling law with up to 7 parameters.

    Features:
      f0 = 1
      f1 = log(non‐vocab parameters + ε)
      f2 = log(num_characters + ε)
      f3 = log(vocab_size + ε)
      f4 = f1^2
      f5 = f2^2
      f6 = f3^2

    Lossu ≈ sum_i params[i] * f_i
    """
    # small epsilon to avoid log(0)
    eps = 1.0
    x1 = np.log(Non_vocab_parameters + eps)
    x2 = np.log(num_characters + eps)
    x3 = np.log(vocab_size + eps)
    # build feature matrix
    F = np.stack([
        np.ones_like(x1),
        x1,
        x2,
        x3,
        x1 * x1,
        x2 * x2,
        x3 * x3
    ], axis=1)  # shape (N,7)
    return F.dot(params)


def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the log‐polynomial scaling law by ordinary least squares.

    Constructs the same 7 features as in scaling_law_func and solves
    for params in min ||F·params − lossu_values||^2 via numpy.linalg.lstsq.
    """
    eps = 1.0
    x1 = np.log(Non_vocab_parameters + eps)
    x2 = np.log(num_characters + eps)
    x3 = np.log(vocab_size + eps)
    F = np.stack([
        np.ones_like(x1),
        x1,
        x2,
        x3,
        x1 * x1,
        x2 * x2,
        x3 * x3
    ], axis=1)  # shape (N,7)

    # solve least squares
    params, *_ = np.linalg.lstsq(F, lossu_values, rcond=None)
    return params


# declare expected number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END