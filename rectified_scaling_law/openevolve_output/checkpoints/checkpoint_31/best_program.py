# EVOLVE-BLOCK-START
"""
Revised scaling law discovery for LLM finetuning scenarios.

We adopt a 4-parameter form:
    loss(x) = a / (x^c + b) + d

Parameters:
  - a > 0 scales the power‐law term
  - b > 0 offsets the denominator for small-x behavior
  - c > 0 is the exponent controlling decay rate
  - d >= 0 is the asymptotic floor as x→∞

This form ensures:
  - loss → d as x → ∞
  - loss ∼ a / x^c + d for large x
  - a finite offset at small x via b
  - Exactly 4 parameters

Fitting procedure:
  - Global search via Differential Evolution
  - Local refinement via L-BFGS-B from multiple starting points
  - Smart heuristic initialization from endpoint fitting
  - Optional user-supplied initial guess
  - Robust bounds and numeric safeguards
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law: loss(x) = a / (x^c + b) + d

    Args:
        data_points: array-like, training sizes x
        params: length-4 array [a, b, c, d]

    Returns:
        predicted losses, same shape as data_points
    """
    x = np.asarray(data_points, dtype=np.float64)
    x = np.maximum(x, 1e-12)
    a, b, c, d = params
    # enforce positivity
    a = np.maximum(a, 1e-12)
    b = np.maximum(b, 1e-12)
    c = np.maximum(c, 1e-12)
    d = np.maximum(d, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.power(x, c) + b
        loss = a / denom + d
    return loss

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the 4-parameter scaling law to data.

    Args:
        data_points: array-like, x values
        loss_values: array-like, observed losses y
        initial_params: optional length-4 initial guess

    Returns:
        best_params: length-4 optimized parameters
    """
    x = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64)
    # remove non-positive y for log fits
    mask = (x > 0) & (y > 0)
    x_fit, y_fit = x[mask], y[mask]

    # bounds: a, b, c, d
    y_max = np.max(y_fit)
    bounds = [
        (1e-8, max(1e-8, y_max * (np.max(x_fit) ** 1))),  # a
        (1e-8, np.max(x_fit) ** 1.5 + 1e2),               # b
        (1e-4,  10.0),                                    # c
        (0.0,   y_max)                                    # d
    ]

    # objective: MSE
    def obj_fn(p):
        preds = scaling_law_func(x, p)
        if not np.all(np.isfinite(preds)):
            return 1e6
        return np.mean((preds - y) ** 2)

    # Heuristic initialization via endpoint power‐law fit
    inits = []
    try:
        d0 = np.min(y_fit) * 0.9
        y0 = np.clip(y_fit - d0, 1e-12, None)
        logx, logy0 = np.log(x_fit), np.log(y0)
        slope, intercept = np.polyfit(logx, logy0, 1)
        c0 = max(1e-3, -slope)
        a0 = max(1e-6, np.exp(intercept))
        b0 = max(1e-6, np.median(np.power(x_fit, c0)))
        inits.append(np.array([a0, b0, c0, d0], dtype=np.float64))
    except Exception:
        pass

    # Include user-provided init if valid
    if initial_params is not None and len(initial_params) == 4:
        inits.append(np.array(initial_params, dtype=np.float64))

    # Global optimization: Differential Evolution
    try:
        de_res = differential_evolution(
            func=obj_fn,
            bounds=bounds,
            strategy='best1bin',
            maxiter=30,
            popsize=15,
            tol=1e-5,
            polish=False,
            seed=42,
            disp=False
        )
        if de_res.success:
            inits.append(de_res.x)
    except Exception:
        pass

    # Add a few random starts around bounds
    rng = np.random.RandomState(2024)
    for _ in range(4):
        rnd = np.array([rng.uniform(lb, ub) for (lb, ub) in bounds], dtype=np.float64)
        inits.append(rnd)

    # Local refinement from all inits
    best_val = np.inf
    best_params = None
    for p0 in inits:
        try:
            res = minimize(
                fun=obj_fn,
                x0=p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol':1e-12, 'gtol':1e-8, 'maxiter':500}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                if best_val < 1e-10:
                    break
        except Exception:
            continue

    # Fallback: if nothing converged, pick heuristic or zeros
    if best_params is None:
        if inits:
            best_params = inits[0]
        else:
            best_params = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)

    # Ensure shape
    best_params = np.asarray(best_params, dtype=np.float64).flatten()[:4]
    return best_params

# Expose parameter count
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Direct testing on CSV data
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    data_sizes = np.array([
        200, 400, 800, 1600, 3200, 6400,
        12800, 25600, 51200, 102400,
        204800, 409600, 819200, 1638400
    ], dtype=np.float64)

    for csv_file in csv_files:
        print(f"\n{'='*60}\nDataset: {csv_file}\n{'='*60}")
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        # assume columns: config name, size, family, then per-size losses
        loss_cols = [c for c in df.columns if c not in ['config name', 'size', 'family']]
        for _, row in df.iterrows():
            model_name = row['config name']
            xs, ys = [], []
            for idx, col in enumerate(loss_cols):
                val = row[col]
                if pd.notna(val) and val > 0:
                    xs.append(data_sizes[idx])
                    ys.append(float(val))
            if len(xs) < 4:
                print(f"Model {model_name}: insufficient data, skipping.")
                continue
            xs_arr, ys_arr = np.array(xs), np.array(ys)
            params = fit_scaling_law(xs_arr, ys_arr)
            preds = scaling_law_func(xs_arr, params)
            mse = np.mean((preds - ys_arr) ** 2)
            print(f"\nModel: {model_name}")
            print(f"  Data points: {len(xs_arr)}; x-range: {xs_arr.min():.0f}–{xs_arr.max():.0f}")
            print(f"  Loss-range: {ys_arr.max():.4f}–{ys_arr.min():.4f}")
            print(f"  Fitted params: {params}")
            print(f"  MSE: {mse:.6e}")
            print(f"  Model size: {int(row['size']):,}, family: {row['family']}")