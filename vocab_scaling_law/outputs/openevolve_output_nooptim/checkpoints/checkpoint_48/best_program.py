# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    A normalized second‐order polynomial in log‐space to capture curvature
    with per‐feature normalization for better conditioning.

    params[0]: bias
    params[1..3]: linear coefficients for each log‐feature
    params[4..6]: quadratic coefficients for each log‐feature
    """
    # take natural logs
    Xv = np.log(vocab_size)
    Xp = np.log(Non_vocab_parameters)
    Xc = np.log(num_characters)

    # compute per‐feature mean and std for normalization
    mv, mp, mc = Xv.mean(), Xp.mean(), Xc.mean()
    sv, sp, sc = Xv.std(), Xp.std(), Xc.std()
    # guard against zero‐variance
    sv = sv if sv > 0 else 1.0
    sp = sp if sp > 0 else 1.0
    sc = sc if sc > 0 else 1.0

    # normalized log‐features
    Lv = (Xv - mv) / sv
    Lp = (Xp - mp) / sp
    Lc = (Xc - mc) / sc

    # polynomial model: bias + linear + quadratic terms
    lossu_pred = (
        params[0]
        + params[1] * Lv
        + params[2] * Lp
        + params[3] * Lc
        + params[4] * (Lv ** 2)
        + params[5] * (Lp ** 2)
        + params[6] * (Lc ** 2)
    )
    return lossu_pred

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the scaling law to the observed Lossu values by minimizing MSE
    via BFGS. Signature and behavior remain unchanged.
    """
    initial_params = np.ones(7)

    def objective(p):
        try:
            pred = scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, p)
            return np.mean((pred - lossu_values) ** 2)
        except:
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# declare expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END