# -*- coding: utf-8 -*-
"""
Unified Evaluator for Scaling Law Discovery.
"""
import argparse
import concurrent.futures
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

# Add parent directory to path to find data_loader
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import load_data

# --- Task Configuration ---
# A set of supported task names. The evaluator will infer which one to use.
SUPPORTED_TASKS = {
    "sft_scaling_law",
    "data_constrained_scaling_law",
    "moe_scaling_law",
    "vocab_scaling_law",
    "domain_mixture_scaling_law",
    "lr_bsz_scaling_law",
    "parallel_scaling_law"
}

# --- Core Functions ---

def get_failure_result(error_msg: str = "Evaluation failed or timed out.") -> Dict[str, Any]:
    """Returns a standardized dictionary for failure cases."""
    return {
        "nmse": 100000.0,
        "nmae": 100000.0,
        "r2": -1.0,
        "combined_score": 0.0,
        "error": error_msg,
    }

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds: int = 600):
    """Runs a function with a specified timeout, raising an exception on timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except Exception as e:
            print(f"Function {func.__name__} timed out or failed: {e}", file=sys.stderr)
            raise

def calculate_final_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculates evaluation metrics, correctly handling multi-dimensional outputs.

    For multi-dimensional targets, metrics (NMSE, NMAE) are calculated for each
    dimension separately and then averaged. The normalization factors (variance
    and mean absolute deviation) are computed using only the test data.

    Args:
        predictions: The model's predictions as a NumPy array.
        true_values: The ground truth values from the test set as a NumPy array.

    Returns:
        A dictionary containing aggregate and per-dimension metrics.
    """
    # 1. Initial validation and type conversion
    try:
        pred = np.asarray(predictions, dtype=float)
        true = np.asarray(true_values, dtype=float)
    except (ValueError, TypeError):
        return get_failure_result("Could not convert predictions or true values to float arrays.")

    # 2. Check for invalid values in predictions
    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("Predictions contain NaN or Inf values.")

    # 3. Reshape 1D arrays to 2D column vectors for consistent processing
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    # 4. Final shape validation
    if true.shape != pred.shape:
        return get_failure_result(f"Shape mismatch: true values {true.shape} vs. predictions {pred.shape}.")
    if true.size == 0:
        return get_failure_result("Cannot evaluate on empty data.")

    # 5. Calculate per-dimension errors on the test set
    test_mse_per_dim = np.mean((true - pred) ** 2, axis=0)
    test_mae_per_dim = np.mean(np.abs(true - pred), axis=0)

    # 6. Calculate normalizers using the test set only
    variance_per_dim = np.var(true, axis=0)
    mean_abs_dev_per_dim = np.mean(np.abs(true - np.mean(true, axis=0)), axis=0)

    # 7. Calculate normalized metrics, avoiding division by zero
    nmse_per_dim = np.divide(test_mse_per_dim, variance_per_dim,
                             out=np.full_like(test_mse_per_dim, np.inf), # Use np.inf where variance is zero
                             where=variance_per_dim > 1e-9)
    nmae_per_dim = np.divide(test_mae_per_dim, mean_abs_dev_per_dim,
                             out=np.full_like(test_mae_per_dim, np.inf), # Use np.inf where MAD is zero
                             where=mean_abs_dev_per_dim > 1e-9)

    # 8. Calculate R^2 for each dimension
    r2_per_dim = 1.0 - nmse_per_dim
    
    # 9. Average per-dimension metrics for final aggregate scores
    nmse = np.mean(nmse_per_dim)
    nmae = np.mean(nmae_per_dim)
    # The standard definition of R^2 relates to the total variance, so it's 1 - (total MSE / total variance)
    # which is equivalent to 1 - mean(nmse_per_dim) if variances are similar, but this is more direct.
    r2 = 1.0 - nmse

    # 10. Compile the results dictionary
    results = {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "combined_score": 1.0 / (1.0 + nmse),
    }

    # 11. Add per-dimension metrics for multi-dimensional targets
    if true.shape[1] > 1:
        results["nmse_per_dim"] = nmse_per_dim.tolist()
        results["nmae_per_dim"] = nmae_per_dim.tolist()
        results["r2_per_dim"] = r2_per_dim.tolist()

    return results


def _import_program(program_path: str):
    """Imports a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from path: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def resolve_task_name(program_path: str) -> str:
    """Infers the task name from environment variables or the file path."""
    env_task = os.getenv("EVAL_TASK_NAME") or os.getenv("SCALING_TASK_NAME")
    if env_task and env_task in SUPPORTED_TASKS:
        return env_task

    p = Path(program_path)
    parts_to_check = [p.parent.name, p.stem]
    for part in parts_to_check:
        for task in SUPPORTED_TASKS:
            if task in part:
                return task

    raise ValueError(
        "Could not resolve task_name. Set env var EVAL_TASK_NAME or "
        f"ensure a supported task name (e.g., '{next(iter(SUPPORTED_TASKS))}') "
        "is in the script's parent folder or file name."
    )

# --- Evaluation Pipelines ---

def evaluate_core(
    program_path: str,
    task_name: str,
    use_test_data: bool = False,
    fitted_params_map: Dict[Any, Any] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Core evaluation logic: fits a model or evaluates it on test data.
    """
    try:
        program = _import_program(program_path)
        fit_scaling_law = program.fit_scaling_law
        scaling_law_func = program.scaling_law_func

        if not use_test_data:
            # --- FIT on training data ---
            train_data = load_data(task_name, train=True)
            if not train_data:
                return get_failure_result("No training data found.")

            new_fitted_params_map = {}
            for key, (X_train, y_train) in train_data.items():
                params = run_with_timeout(fit_scaling_law, args=(X_train, y_train))
                new_fitted_params_map[key] = params
            return {"fitted_params": new_fitted_params_map}

        else:
            # --- EVALUATE on test data ---
            if fitted_params_map is None:
                return get_failure_result("fitted_params_map is required for evaluation.")

            test_data = load_data(task_name, train=False)
            if not test_data:
                return get_failure_result("No test data found.")

            all_predictions, all_true_values = [], []
            for key, (X_test, y_test) in test_data.items():
                if key not in fitted_params_map:
                    print(f"Warning: No params for test group '{key}'. Skipping.", file=sys.stderr)
                    continue

                params = fitted_params_map[key]
                predictions = run_with_timeout(scaling_law_func, args=(X_test, params))
                all_predictions.append(np.asarray(predictions))
                all_true_values.append(np.asarray(y_test))

            if not all_predictions:
                return get_failure_result("No predictions were generated for the test set.")

            final_predictions = np.concatenate(all_predictions)
            final_true_values = np.concatenate(all_true_values)

            return calculate_final_metrics(
                final_predictions,
                final_true_values,
            )

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result(str(e))

def evaluate(program_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    High-level, single-call evaluation function.

    This orchestrates the entire process:
    1. Infers the task name.
    2. Fits the model on training data.
    3. Evaluates the fitted model on test data.
    4. Returns a dictionary with final metrics and (optionally) fitted parameters.

    Args:
        program_path: Path to the user's Python script with scaling law functions.
        verbose: If True, include fitted parameters and task name in the result.

    Returns:
        A dictionary containing the evaluation results.
    """
    try:
        task_name = resolve_task_name(program_path)
    except ValueError as e:
        return get_failure_result(str(e))

    # 1. Fit on training data to get parameters
    fit_result = evaluate_core(program_path, task_name, use_test_data=False)
    if "fitted_params" not in fit_result:
        error = fit_result.get("error", "Unknown fitting error.")
        return get_failure_result(f"Fitting failed: {error}")

    fitted_params_map = fit_result["fitted_params"]

    # 2. Evaluate on test data using the fitted parameters
    test_result = evaluate_core(
        program_path,
        task_name,
        use_test_data=True,
        fitted_params_map=fitted_params_map,
    )

    # 3. Combine results into a comprehensive output
    if verbose:
        test_result["fitted_params"] = fitted_params_map
        test_result["task_name"] = task_name
    return test_result

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluator for Scaling Law Discovery.")
    parser.add_argument("program_path", type=str, help="Path to the Python script with scaling law functions.")
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Path '{args.program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running Evaluation for Program: {args.program_path} ---")
    final_results = evaluate(args.program_path, verbose=True)

    task_name = final_results.get('task_name', 'N/A')
    print(f"Inferred Task: {task_name}")

    if "error" in final_results and final_results["error"]:
        print("\n--- ⛔ EVALUATION FAILED ⛔ ---")
        print(f"Error: {final_results['error']}")
        sys.exit(1)

    print("\n--- ✅ Final Test Results (Aggregate) ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  R-squared (R²):        {final_results.get('r2', 'N/A'):.6f}")
    print(f"  Combined Score:        {final_results.get('combined_score', 'N/A'):.6f}")
    
    # Print per-dimension metrics if they exist
    if "nmse_per_dim" in final_results:
        print("\n  --- Per-Dimension Metrics ---")
        nmse_vals = final_results["nmse_per_dim"]
        nmae_vals = final_results["nmae_per_dim"]
        r2_vals = final_results["r2_per_dim"]
        for i, (nmse_d, nmae_d, r2_d) in enumerate(zip(nmse_vals, nmae_vals, r2_vals)):
            print(f"    Dim {i+1}: NMSE={nmse_d:.4f}, NMAE={nmae_d:.4f}, R²={r2_d:.4f}")

    params = final_results.get('fitted_params', {})
    if params:
        print(f"\nFitted parameters for {len(params)} group(s):")
        for key, val in params.items():
            param_val = np.asarray(val)
            if param_val.size > 1:
                param_str = np.array2string(param_val, precision=4, max_line_width=80, suppress_small=True)
            else:
                param_str = f"{param_val.item():.4f}" # Use .item() for single-element arrays
            print(f"  - Group '{key}': {param_str}")
    print("--------------------------")