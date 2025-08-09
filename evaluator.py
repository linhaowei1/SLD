"""
Unified Evaluator for Scaling Law Discovery.

This evaluator is refactored to work with a standardized data format where all
input features are consolidated into a single NumPy array `X`.

It supports two calling styles:
1. Task-specific (recommended): `evaluate(program_path)`
   - Infers the task from the environment or path.
   - Automatically runs the fit -> evaluate pipeline.
2. Core/Generic: `evaluate_core(program_path, task_name, ...)`
   - Requires explicit task name and manual control over fitting vs. evaluating.
"""
import numpy as np
import os
import sys
import traceback
import concurrent.futures
import importlib.util
from pathlib import Path
from typing import Dict, Any, Union

# Add parent directory to path to find the new data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

# --- Task Configuration ---
# This defines the feature order for each task's X matrix.
TASK_CONFIG = {
    "rectified_scaling_law": {"scaling_vars": ["data_size"], "response_var": "loss_values"},
    "data_constrained_scaling_law": {"scaling_vars": ["tokens", "params", "unique_tokens"], "response_var": "loss"},
    "moe_scaling_law": {"scaling_vars": ["num_experts", "dense_parameter_count"], "response_var": "loss_validation"},
    "vocab_scaling_law": {"scaling_vars": ["Non_vocab_parameters", "vocab_size", "num_characters"], "response_var": "Lossu"},
    "domain_mixture_scaling_law": {"scaling_vars": ["proportions"], "response_var": "loss"},
    "lr_scaling_law": {"scaling_vars": ['lr', 'bsz', 'data_size', 'non_embedding_param_size'], "response_var": "lm loss"}
}


def get_failure_result(error_msg: str = "Evaluation failed or timed out.") -> Dict[str, float]:
    """Returns a standardized dictionary for failure cases."""
    return {
        "nmse": 100000.0,
        "nmae": 100000.0,
        "r2": -1.0,
        "combined_score": 0.0,
        "error": error_msg,
    }

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=600):
    """Runs a function with a specified timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except Exception as e:
            print(f"Function {func.__name__} timed out or failed: {e}", file=sys.stderr)
            raise

def calculate_final_metrics(predictions: np.ndarray, true_values: np.ndarray) -> Dict[str, float]:
    """Calculates final evaluation metrics: NMSE, NMAE, R^2."""
    pred = np.asarray(predictions, dtype=float).flatten()
    true = np.asarray(true_values, dtype=float).flatten()

    valid_mask = ~(np.isnan(pred) | np.isinf(pred))
    if not np.any(valid_mask) or pred.shape != true.shape:
        return get_failure_result("Prediction failed, is invalid, or has shape mismatch.")

    pred_filtered, true_filtered = pred[valid_mask], true[valid_mask]
    if len(pred_filtered) == 0:
        return get_failure_result("No valid data points left after filtering.")

    test_mse = np.mean((true_filtered - pred_filtered) ** 2)
    test_mae = np.mean(np.abs(true_filtered - pred_filtered))
    
    # Variance is calculated on the true test values, as is standard
    total_variance = np.var(true_filtered)
    total_mean_abs_deviation = np.mean(np.abs(true_filtered - np.mean(true_filtered)))
    
    nmse = test_mse / total_variance if total_variance > 1e-9 else test_mse
    nmae = test_mae / total_mean_abs_deviation if total_mean_abs_deviation > 1e-9 else test_mae
    r2 = 1.0 - nmse
    
    return {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "combined_score": 1.0 / (1.0 + nmse),
    }

def _import_program(program_path: str):
    """Imports a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def resolve_task_name(program_path: str) -> str:
    """Infers the task name from environment variables or file path."""
    env_task = os.getenv("EVAL_TASK_NAME") or os.getenv("SCALING_TASK_NAME")
    if env_task and env_task in TASK_CONFIG:
        return env_task
    
    p = Path(program_path)
    # Check parent directory name, then file name
    parts_to_check = [p.parent.name, p.stem]
    for part in parts_to_check:
        for key in TASK_CONFIG:
            if key in part:
                return key

    raise ValueError(
        "Could not resolve task_name. Set env var EVAL_TASK_NAME or "
        f"ensure the task name (e.g., '{list(TASK_CONFIG.keys())[0]}') "
        "is in the script's parent folder or file name."
    )

def evaluate_core(
    program_path: str,
    task_name: str,
    use_test_data: bool = False,
    fitted_params_map: Dict[Any, Any] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Core evaluation logic. Handles fitting or predicting based on `use_test_data`.
    """
    try:
        program = _import_program(program_path)
        fit_scaling_law = program.fit_scaling_law
        scaling_law_func = program.scaling_law_func

        if not use_test_data:
            # --- FIT on training data ---
            train_data = load_data(task_name, train=True)
            if not train_data:
                return {"error": "No training data found."}

            new_fitted_params_map = {}
            for key, (X_train, y_train) in train_data.items():
                params = run_with_timeout(fit_scaling_law, args=(X_train, y_train))
                new_fitted_params_map[key] = params
            return {"fitted_params": new_fitted_params_map}

        else:
            # --- EVALUATE on test data ---
            if fitted_params_map is None:
                return {"error": "fitted_params_map is required for evaluation."}

            test_data = load_data(task_name, train=False)
            if not test_data:
                return {"error": "No test data found."}

            all_predictions, all_true_values = [], []
            for key, (X_test, y_test) in test_data.items():
                if key not in fitted_params_map:
                    print(f"Warning: No params for test group {key}. Skipping.", file=sys.stderr)
                    continue
                
                params = fitted_params_map[key]
                predictions = run_with_timeout(scaling_law_func, args=(X_test, params))
                all_predictions.append(np.asarray(predictions).flatten())
                all_true_values.append(np.asarray(y_test).flatten())

            if not all_predictions:
                return get_failure_result("No predictions were generated for the test set.")

            final_predictions = np.concatenate(all_predictions)
            final_true_values = np.concatenate(all_true_values)
            
            return calculate_final_metrics(final_predictions, final_true_values)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result()

def evaluate(program_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    High-level, single-argument evaluation function.
    
    This function orchestrates the entire process:
    1. Infers the task name from the environment or file path.
    2. Calls `evaluate_core` to fit the model on training data.
    3. Calls `evaluate_core` again to evaluate the fitted model on test data.
    4. Returns a dictionary with final metrics and the fitted parameters.

    Args:
        program_path: The path to the user's Python script with scaling law functions.

    Returns:
        A dictionary containing the evaluation results.
    """
    try:
        task_name = resolve_task_name(program_path)
    except ValueError as e:
        return get_failure_result(str(e))

    # 1. Fit on training data
    fit_result = evaluate_core(program_path, task_name, use_test_data=False)
    if "fitted_params" not in fit_result:
        error = fit_result.get("error", "Unknown fitting error.")
        return get_failure_result(f"Fitting failed: {error}")

    fitted_params_map = fit_result["fitted_params"]

    # 2. Evaluate on test data
    test_result = evaluate_core(
        program_path,
        task_name,
        use_test_data=True,
        fitted_params_map=fitted_params_map,
    )
    
    # Combine results for a comprehensive output
    if verbose:
        test_result["fitted_params"] = fitted_params_map
        test_result["task_name"] = task_name
    return test_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Evaluator for Scaling Law Discovery.")
    parser.add_argument("program_path", type=str, help="Path to the Python script with scaling law functions.")
    
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Path '{args.program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    print(f"--- Running Evaluation for Program: {args.program_path} ---")

    # Use the simple, high-level evaluate function
    final_results = evaluate(args.program_path, verbose=True)
    
    task_name = final_results.get('task_name', 'N/A')
    print(f"Inferred Task: {task_name}")

    if "error" in final_results and final_results["error"]:
        print("\n--- EVALUATION FAILED ---")
        print(f"Error: {final_results['error']}")
        sys.exit(1)
    
    print("\n--- Final Test Results ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  R-squared (R2):        {final_results.get('r2', 'N/A'):.6f}")
    print(f"  Combined Score:        {final_results.get('combined_score', 'N/A'):.6f}")
    
    params = final_results.get('fitted_params', {})
    print(f"\nFitted parameters for {len(params)} group(s):")
    for key, val in params.items():
        # Truncate long parameter arrays for cleaner printing
        param_str = np.array2string(val, precision=4, max_line_width=80, suppress_small=True)
        print(f"  - Group '{key}': {param_str}")
    print("--------------------------")