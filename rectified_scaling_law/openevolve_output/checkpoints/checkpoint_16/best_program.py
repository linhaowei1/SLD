# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios.

We adopt a 4-parameter form:
    loss(x) = a / (x^c + b) + d

This captures:
  - As x → ∞, loss → d (irreducible floor)
  - Power‐law decay with exponent c and offset b
  - Only 4 parameters for interpretability and efficiency

Parameter constraints / transformations:
  a ≥ 0, b ≥ 0, c ≥ 1e-8, d ≥ 0

We fit via a multi-start L-BFGS-B routine with a data-driven seed
and several random restarts, choosing the lowest‐MSE solution.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law: loss = a / (x^c + b) + d

    Args:
      data_points: array_like, training data sizes
      params: array_like of length 4, raw parameters [p0, p1, p2, p3]

    Returns:
      np.ndarray of predicted losses
    """
    p = np.asarray(params, dtype=float).ravel()
    # Ensure exactly 4 entries
    if p.size < 4:
        p = np.pad(p, (0, 4 - p.size), 'constant')
    elif p.size > 4:
        p = p[:4]

    eps = 1e-12
    # Enforce non-negativity / stability
    a = np.maximum(p[0], 0.0) + eps
    b = np.maximum(p[1], 0.0) + eps
    c = np.maximum(p[2], eps)
    d = np.maximum(p[3], 0.0)

    x = np.asarray(data_points, dtype=float)
    x = np.maximum(x, eps)

    # Compute loss
    # loss = a / (x**c + b) + d
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        loss = a / (np.power(x, c) + b) + d

    # Replace non-finite entries
    if not np.all(np.isfinite(loss)):
        finite = loss[np.isfinite(loss)]
        fallback = finite.max() if finite.size else eps
        loss = np.where(np.isfinite(loss), loss, fallback)

    return loss

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the 4-parameter scaling law using multi-start L-BFGS-B.

    Args:
      data_points: array_like, training data sizes
      loss_values: array_like, observed losses
      initial_params: optional array_like of length 4 to seed one start

    Returns:
      np.ndarray of optimized raw parameters (length 4)
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Objective: mean squared error
    def mse_obj(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2)

    # Bounds: a>=0, b>=0, c>=1e-8, d>=0
    bounds = [(0.0, None), (0.0, None), (1e-8, 5.0), (0.0, None)]

    # Generate initial guesses
    y_min, y_max = np.min(y), np.max(y)
    x_max = np.max(x)

    # 1) data-driven guess
    c0 = 0.5
    b0 = 1.0
    d0 = y_min
    a0 = max((y_max - d0) * (b0 + 1.0), eps := 1e-8)
    seeds = [np.array([a0, b0, c0, d0], dtype=float)]

    # 2) user-provided initial
    if initial_params is not None:
        arr = np.asarray(initial_params, dtype=float).ravel()
        if arr.size == 4:
            seeds.append(arr)

    # 3) random restarts
    rng = np.random.RandomState(0)
    for _ in range(7):
        a_r = rng.uniform(0.1 * (y_max - y_min + eps), 10 * (y_max - y_min + eps))
        b_r = rng.uniform(1e-2, max(1.0, x_max**0.5))
        c_r = rng.uniform(0.1, 3.0)
        d_r = rng.uniform(0.0, y_max)
        seeds.append(np.array([a_r, b_r, c_r, d_r], dtype=float))

    best_loss = np.inf
    best_params = None

    # Run L-BFGS-B from each seed
    for seed in seeds:
        res = minimize(mse_obj, seed, method='L-BFGS-B',
                       bounds=bounds,
                       options={'ftol':1e-9, 'gtol':1e-8, 'maxiter':1000})
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # Fallback: if none succeeded, use the best seed
    if best_params is None:
        best_params = seeds[0]

    # Ensure exactly 4 params
    best = np.asarray(best_params, dtype=float).ravel()
    if best.size < 4:
        best = np.pad(best, (0, 4 - best.size), 'constant')
    elif best.size > 4:
        best = best[:4]

    return best

# Expose parameter count
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import pandas as pd
    import os

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
        loss_columns = df.columns[1:-2]

        for _, row in df.iterrows():
            config = row['config name']
            raw_losses = []
            raw_sizes = []
            for i, col in enumerate(loss_columns[1:], start=1):
                v = row[col]
                if pd.notna(v) and v > 0:
                    raw_losses.append(float(v))
                    raw_sizes.append(data_sizes[i-1])

            if len(raw_sizes) < 4:
                print(f"Model {config}: <4 points, skipping.")
                continue

            xs = np.array(raw_sizes, dtype=float)
            ys = np.array(raw_losses, dtype=float)

            print(f"\nModel: {config}")
            print(f"  Data pts: {len(xs)}, size range: {xs.min()}–{xs.max()}, "
                  f"loss range: {ys.max():.4f}–{ys.min():.4f}")

            params = fit_scaling_law(xs, ys)
            preds = scaling_law_func(xs, params)
            mse = float(np.mean((preds - ys) ** 2))

            print(f"  Params: {params}")
            print(f"  MSE: {mse:.6f}")
            print(f"  Model size: {int(row['size']):,}, Family: {row['family']}")