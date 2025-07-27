# EVOLVE-BLOCK-START
"""
Evolved vocab-based scaling law for LLM training scenarios.

We model Lossu as a linear function of log‐features and
their pairwise interactions. This yields a closed-form
least‐squares fit (7 parameters) that is fast, stable, and
interpretable.

Scaling Law:
    Lv = log1p(vocab_size)
    Lp = log1p(Non_vocab_parameters)
    Lc = log1p(num_characters)

    Lossu ≈ p0
          + p1*Lv
          + p2*Lp
          + p3*Lc
          + p4*(Lv*Lp)
          + p5*(Lv*Lc)
          + p6*(Lp*Lc)

Features: [1, Lv, Lp, Lc, Lv*Lp, Lv*Lc, Lp*Lc]
"""
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu given data arrays and parameter vector of length 7.
    """
    p0, p1, p2, p3, p4, p5, p6 = params
    # compute log‐features
    Lv = np.log1p(vocab_size)
    Lp = np.log1p(Non_vocab_parameters)
    Lc = np.log1p(num_characters)
    # linear model with interactions
    return (
        p0
        + p1 * Lv
        + p2 * Lp
        + p3 * Lc
        + p4 * (Lv * Lp)
        + p5 * (Lv * Lc)
        + p6 * (Lp * Lc)
    )

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7 parameters of the log‐feature linear scaling law via OLS.
    Returns a length‐7 array of fitted parameters.
    """
    # ensure numpy arrays
    NVP = np.asarray(Non_vocab_parameters, dtype=np.float64)
    VS  = np.asarray(vocab_size,         dtype=np.float64)
    NC  = np.asarray(num_characters,     dtype=np.float64)
    Y   = np.asarray(lossu_values,       dtype=np.float64)

    # compute features
    Lv = np.log1p(VS)
    Lp = np.log1p(NVP)
    Lc = np.log1p(NC)

    # Design matrix columns
    # col0: intercept
    # col1: Lv
    # col2: Lp
    # col3: Lc
    # col4: Lv * Lp
    # col5: Lv * Lc
    # col6: Lp * Lc
    X = np.vstack([
        np.ones_like(Lv),
        Lv,
        Lp,
        Lc,
        Lv * Lp,
        Lv * Lc,
        Lp * Lc
    ]).T  # shape (n_samples, 7)

    # Solve linear least squares
    # params: shape (7,)
    params, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return params

# annotate expected number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END