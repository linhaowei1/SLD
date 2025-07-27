# EVOLVE-BLOCK-START
"""
Vocab scaling law discovery for LLM training scenarios
Evolved version: uses a log‐feature linear model with pairwise interactions
for stability, interpretability, and fast closed‐form fitting.
"""
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu via a linear model on log‐features and their pairwise products.
    Uses exactly 7 parameters.

    Args:
        Non_vocab_parameters: array_like, shape (n,)
        vocab_size:            array_like, shape (n,)
        num_characters:        array_like, shape (n,)
        params:                array_like, shape (7,)

    Returns:
        lossu_pred: np.ndarray, shape (n,)
    """
    # convert to numpy arrays
    N = np.asarray(Non_vocab_parameters, dtype=float)
    V = np.asarray(vocab_size,         dtype=float)
    C = np.asarray(num_characters,     dtype=float)

    # small constant to avoid log(0)
    eps = 1e-8
    lnN = np.log(N + eps)
    lnV = np.log(V + eps)
    lnC = np.log(C + eps)

    # build feature matrix: [1, lnN, lnV, lnC, lnN*lnV, lnN*lnC, lnV*lnC]
    # each row i: features for sample i
    X0 = np.ones_like(lnN)
    X1 = lnN
    X2 = lnV
    X3 = lnC
    X4 = lnN * lnV
    X5 = lnN * lnC
    X6 = lnV * lnC

    # stack and multiply by params
    # shape (n,7) dot (7,) -> (n,)
    X = np.stack([X0, X1, X2, X3, X4, X5, X6], axis=1)
    return X.dot(params)


def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter scaling law by linear least squares
    on log‐features and their pairwise interactions.

    Args:
        Non_vocab_parameters: array_like, shape (n,)
        vocab_size:            array_like, shape (n,)
        num_characters:        array_like, shape (n,)
        lossu_values:          array_like, shape (n,)

    Returns:
        params: np.ndarray, shape (7,)
    """
    # convert inputs
    N = np.asarray(Non_vocab_parameters, dtype=float)
    V = np.asarray(vocab_size,         dtype=float)
    C = np.asarray(num_characters,     dtype=float)
    y = np.asarray(lossu_values,       dtype=float)

    # avoid log(0)
    eps = 1e-8
    lnN = np.log(N + eps)
    lnV = np.log(V + eps)
    lnC = np.log(C + eps)

    # build design matrix X (n_samples x 7)
    X0 = np.ones_like(lnN)
    X1 = lnN
    X2 = lnV
    X3 = lnC
    X4 = lnN * lnV
    X5 = lnN * lnC
    X6 = lnV * lnC
    X = np.stack([X0, X1, X2, X3, X4, X5, X6], axis=1)

    # solve for params by least squares: minimize ||X·p - y||^2
    # use small ridge for numerical stability
    ridge = 1e-6
    # normal equations: (X^T X + ridge*I) p = X^T y
    XT_X = X.T.dot(X)
    diag = np.eye(XT_X.shape[0]) * ridge
    lhs = XT_X + diag
    rhs = X.T.dot(y)

    # solve linear system
    try:
        params = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        # fallback to lstsq
        params, *_ = np.linalg.lstsq(X, y, rcond=None)

    return params


# annotate expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END