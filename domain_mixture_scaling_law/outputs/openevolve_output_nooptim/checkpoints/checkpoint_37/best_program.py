"""
Domain‐mixture scaling law discovery for LLM training scenarios

Enhanced mixture‐exponential form with domain‐specific sensitivities:
    L_i(r) = c_i + k_i * exp( ∑_{j=1}^5 t_{ij} * r_j )

Where for each domain i=1..5:
  - c_i : domain‐specific bias (5 parameters)
  - k_i : domain‐specific scale (5 parameters, enforced positive via exp of raw)
  - t_{ij} : domain‐i sensitivity to mixture proportion of domain j (5×5=25 parameters)

Total parameters: 35 = (5 c_i) + (5 raw log-k_i) + (25 t_{ij})
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    Vectorized domain‐mixture exponential scaling law:
        L_i(r) = c_i + k_i * exp(∑_j t_{ij} * r_j)

    Params layout (35 total):
      params[0:5]    = c_i        (bias per domain, real)
      params[5:10]   = logk_i     (raw log-scale per domain, real)
      params[10:35]  = t_matrix   (25 raw sensitivities, real),
                        row-major for t_{i0}..t_{i4}

    Args:
        proportions: array shape [n_samples, 5], each row sums to 1
        params:      array of length ≥35 (will be padded/truncated)
    Returns:
        loss_predictions: array shape [n_samples, 5]
    """
    # ensure 2D input
    proportions = np.atleast_2d(proportions).astype(float)
    # flatten & pad/truncate parameter vector to length 35
    p = np.asarray(params, dtype=float).ravel()
    if p.size < 35:
        p = np.concatenate([p, np.zeros(35 - p.size, dtype=float)])
    else:
        p = p[:35]
    # unpack parameters
    c = p[0:5]                    # biases
    logk = p[5:10]                # raw log-scales
    t_flat = p[10:35]             # 25 sensitivities
    k = np.exp(logk)              # enforce k_i ≥ 0
    # reshape sensitivities to (5 domains × 5 mixture dims)
    t_mat = t_flat.reshape(5, 5)
    # compute per‐sample per‐domain exponent argument
    # mix_dot[sample, i] = sum_j proportions[sample, j] * t_{ij}
    mix_dot = proportions.dot(t_mat.T)  # shape (n_samples, 5)
    exp_term = np.exp(mix_dot)          # shape (n_samples, 5)
    # apply scale and bias
    loss_predictions = exp_term * k[None, :] + c[None, :]
    return loss_predictions

# specify expected number of parameters
scaling_law_func.num_params = 35

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law to domain proportions and loss values using BFGS.

    Args:
        proportions: array [n_samples, 5] of domain proportions
        loss_values: array [n_samples, 5] of observed losses

    Returns:
        best_params: array of length 35 (c_i, logk_i, t_{ij})
    """
    # number of params dictated by scaling_law_func
    N = scaling_law_func.num_params
    # initialize parameters
    initial_params = np.ones(N, dtype=float)
    # c_i initial = mean loss per domain
    for i in range(5):
        initial_params[i] = np.mean(loss_values[:, i])
        initial_params[5 + i] = 0.0         # logk_i = 0 => k_i=1
    # t_{ij} initial = 1.0 for all
    initial_params[10:] = 1.0

    def objective(params):
        try:
            preds = scaling_law_func(proportions, params)
            # mean squared error
            return np.mean((preds - loss_values) ** 2)
        except:
            return 1e6

    best_params = initial_params.copy()
    best_loss = float('inf')
    # three random restarts
    for attempt in range(3):
        if attempt == 0:
            start = initial_params
        else:
            start = initial_params + np.random.randn(N) * 0.05
        try:
            res = minimize(objective, start, method='BFGS')
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except:
            pass

    return best_params