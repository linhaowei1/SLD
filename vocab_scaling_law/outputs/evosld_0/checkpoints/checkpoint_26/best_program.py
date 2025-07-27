# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu via a 7-parameter log-linear model with pairwise interactions:
      features = [1,
                  log(P+1),
                  log(V+1),
                  log(N+1),
                  log(P+1)*log(V+1),
                  log(P+1)*log(N+1),
                  log(V+1)*log(N+1)]
      Lossu = params · features
    """
    P = np.asarray(Non_vocab_parameters, dtype=float)
    V = np.asarray(vocab_size,         dtype=float)
    N = np.asarray(num_characters,     dtype=float)
    # compute stabilized logs
    lp = np.log(P + 1.0)
    lv = np.log(V + 1.0)
    ln = np.log(N + 1.0)
    # pairwise interaction terms
    f_pv = lp * lv
    f_pn = lp * ln
    f_vn = lv * ln
    # stack into feature matrix
    X = np.stack([
        np.ones_like(lp),
        lp,
        lv,
        ln,
        f_pv,
        f_pn,
        f_vn
    ], axis=1)  # shape (n_samples, 7)
    return X.dot(params)


def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter log-linear scaling law by ordinary least squares.
    Returns an array of 7 parameters that minimize ||X·params - y||².
    """
    P = np.asarray(Non_vocab_parameters, dtype=float)
    V = np.asarray(vocab_size,         dtype=float)
    N = np.asarray(num_characters,     dtype=float)
    y = np.asarray(lossu_values,       dtype=float)

    # build the same feature matrix as in scaling_law_func
    lp = np.log(P + 1.0)
    lv = np.log(V + 1.0)
    ln = np.log(N + 1.0)
    f_pv = lp * lv
    f_pn = lp * ln
    f_vn = lv * ln
    X = np.stack([
        np.ones_like(lp),
        lp,
        lv,
        ln,
        f_pv,
        f_pn,
        f_vn
    ], axis=1)

    # solve for params via least squares
    # params shape = (7,)
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    return params

# annotate expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END