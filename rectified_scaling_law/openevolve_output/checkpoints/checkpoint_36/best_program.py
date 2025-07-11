# EVOLVE-BLOCK-START
"""
Evolved 4-parameter scaling law for LLM finetuning:

    loss(x) = a / (x^b + c) + d

where:
  a > 0    : controls initial offset
  b > 0    : decay exponent
  c > 0    : horizontal shift in denominator
  d >= 0   : asymptotic loss floor

We fit by optimizing the log of parameters (to enforce positivity)
using multi-start L-BFGS-B on a robust MSE objective.
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    4-parameter hyperbolic-power-law:
        loss(x) = a / (x^b + c) + d

    Args:
        data_points: array-like of training sizes (x > 0)
        params: array-like of 4 parameters [a, b, c, d]
    Returns:
        numpy array of predicted losses
    """
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 4:
        p = np.resize(p, 4)
    a, b, c, d = p
    x = np.asarray(data_points, dtype=float)
    # avoid zero or negative x
    x = np.maximum(x, 1e-12)
    # compute x^b robustly
    xb = np.exp(b * np.log(x))
    return a / (xb + c) + d

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the 4-parameter law: loss = a/(x^b + c) + d
    by optimizing log-parameters with multi-start L-BFGS-B.
    """
    # prepare and clean data
    x = np.asarray(data_points, dtype=float).ravel()
    y = np.asarray(loss_values, dtype=float).ravel()
    valid = (x > 0) & (y >= 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 4:
        # not enough data: return trivial defaults
        return np.array([1.0, 0.5, 1.0, 0.0], dtype=float)

    # Shifted y for fitting exponent
    y_min = y.min()
    y_shift = y - y_min + 1e-8

    # Initial guess via log-log linear fit for b
    try:
        coef = np.polyfit(np.log(x), np.log(y_shift), 1)
        slope = coef[0]
        b0 = float(-slope)
        # constrain b0
        b0 = np.clip(b0, 1e-3, 10.0)
    except Exception:
        b0 = 0.5

    # initial d ~= y_min * 0.9 (floor)
    d0 = max(y_min * 0.9, 1e-8)
    # initial c at median(x)^b0
    c0 = float(np.median(x) ** b0)
    # initial a by matching at x_min: a/(x_min^b0 + c0) + d0 ~= y_max
    y_max = y.max()
    x_min = x.min()
    denom0 = x_min**b0 + c0
    a0 = max((y_max - d0) * denom0, 1e-8)

    init_params = np.array([a0, b0, c0, d0], dtype=float)
    # override if user gives a valid initial_params
    if initial_params is not None:
        ip = np.asarray(initial_params, dtype=float).ravel()
        if ip.size == 4 and np.all(np.isfinite(ip)) and (ip[0] > 0) and (ip[1] > 0) and (ip[2] > 0):
            init_params = ip.copy()

    # work in log-space for positivity
    log_init = np.log(init_params + 1e-12)

    # objective: mean squared error in original space
    def obj_logp(logp):
        p = np.exp(logp)
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2)

    # bounds on raw log-parameters:
    #   log(a): [-20, 20], log(b): [-5, 5], log(c): [-20,20], log(d): [-20,20]
    bounds = [(-20, 20), (-5, 5), (-20, 20), (-20, 20)]

    # generate multi-start inits
    rng = np.random.RandomState(0)
    inits = [log_init]
    # jittered around main init
    for _ in range(5):
        noise = rng.normal(scale=0.5, size=4)
        cand = log_init + noise
        # clip to bounds
        for i, (lo, hi) in enumerate(bounds):
            cand[i] = np.clip(cand[i], lo, hi)
        inits.append(cand)
    # one fully random start
    rand_start = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
    inits.append(rand_start)

    best_val = np.inf
    best_logp = None
    for lp0 in inits:
        try:
            res = minimize(
                obj_logp, lp0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_logp = res.x
        except Exception:
            continue

    # if no run succeeded, fallback to init
    if best_logp is None:
        best_logp = log_init

    # return actual parameters
    best_params = np.exp(best_logp)
    # ensure shape
    best_params = np.asarray(best_params, dtype=float).ravel()
    if best_params.size != 4:
        best_params = np.resize(best_params, 4)
    return best_params

# advertise parameter count
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    data_sizes = np.array([
        200, 400, 800, 1600, 3200, 6400,
        12800, 25600, 51200, 102400,
        204800, 409600, 819200, 1638400
    ], dtype=float)

    for csv_file in csv_files:
        print(f"\n{'='*50}\nProcessing dataset: {csv_file}\n{'='*50}")
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        loss_cols = [c for c in df.columns if c not in ('config name','size','family')]

        for _, row in df.iterrows():
            model_name = row['config name']
            sizes, losses = [], []
            for i, col in enumerate(loss_cols):
                if i < len(data_sizes):
                    y = row[col]
                    if pd.notna(y) and y >= 0:
                        sizes.append(data_sizes[i])
                        losses.append(float(y))
            if len(sizes) < 4:
                print(f"Model {model_name}: insufficient data, skipping.")
                continue

            sizes = np.array(sizes, dtype=float)
            losses = np.array(losses, dtype=float)
            print(f"\nModel: {model_name}")
            print(f"Data sizes: {sizes.min()} - {sizes.max()} ({len(sizes)} points)")
            print(f"Losses: {losses.max():.4f} -> {losses.min():.4f}")

            params = fit_scaling_law(sizes, losses)
            preds = scaling_law_func(sizes, params)
            mse = np.mean((preds - losses) ** 2)

            print(f"Fitted params: {params}")
            print(f"Mean squared error: {mse:.6f}")
            print(f"Model size: {int(row['size']):,} parameters")
            print(f"Model family: {row['family']}")