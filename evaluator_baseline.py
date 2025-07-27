"""
Generalized Evaluator for Scaling Law Discovery Programs (V3 - No Control Variables)

This version supports all tasks from TASK_CONFIG where "control_vars" is empty.
It dynamically loads data and calls the user program based on the task name.

Core Logic:
1. Receives `task_name` and `program_path` from command line.
2. Uses `TASK_CONFIG` to determine scaling variables and response variable for the task.
3. Dynamically prepares data and passes them as separate parameters to the user's `discover_scaling_law` function.
4. Uses the same NMSE/NMAE metrics as V2 for evaluation.
"""

import importlib.util
import numpy as np
import os
import sys
import traceback
from typing import Dict, Any, Tuple, List

# Add parent directory to path for data_loader import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

# --- Task Configuration ---
TASK_CONFIG = {
    "data_constrained_scaling_law": {
        "scaling_vars": ["data_size", "model_size", "unique_tokens"],
        "control_vars": [],
        "response_var": "loss_values",
    },
    "moe_scaling_law": {
        "scaling_vars": ["total_parameter_count", "num_experts"],
        "control_vars": [],
        "response_var": "loss_values",
    },
    "vocab_scaling_law": {
        "scaling_vars": ["Non_vocab_parameters", "vocab_size", "num_characters"],
        "control_vars": [],
        "response_var": "lossu_values",
    },
}

def get_failure_result() -> Dict[str, Any]:
    """
    Return a standard result for failure cases, ensuring the same key structure as success cases.
    """
    worst_nmse = 100000.0
    worst_nmae = 100000.0
    worst_r2 = -1.0
    worst_score = 1.0 / (1.0 + worst_nmse)

    result = {
        "nmse": worst_nmse,
        "nmae": worst_nmae,
        "r2": worst_r2,
        "combined_score": worst_score,
        "equation": "Failed to discover or evaluate the equation.",
        "error": "Evaluation failed or user program raised an exception.",
    }
    return result

def calculate_final_metrics(
    predicted_values: np.ndarray,
    true_values: np.ndarray,
    total_response_values: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate fit quality and calculate normalized metrics based on total dataset statistics.
    """
    pred = np.asarray(predicted_values, dtype=float).flatten()
    true = np.asarray(true_values, dtype=float).flatten()
    total_y = np.asarray(total_response_values, dtype=float).flatten()

    if pred.shape != true.shape:
        return {"error": "Predicted and true values shape mismatch."}

    valid_mask = ~(np.isnan(pred) | np.isinf(pred) | np.isnan(true) | np.isinf(true))
    if not np.any(valid_mask):
        return {"error": "No valid data points to compare after filtering."}

    pred_filtered = pred[valid_mask]
    true_filtered = true[valid_mask]

    if len(pred_filtered) < 1:
        return {"error": "Insufficient valid data points for evaluation."}

    test_mse = np.mean((true_filtered - pred_filtered) ** 2)
    test_mae = np.mean(np.abs(true_filtered - pred_filtered))

    total_variance = np.var(total_y)
    total_mean_abs_deviation = np.mean(np.abs(total_y - np.mean(total_y)))

    nmse = test_mse / total_variance if total_variance > 0 else test_mse
    nmae = test_mae / total_mean_abs_deviation if total_mean_abs_deviation > 0 else test_mae

    r2 = 1.0 - nmse
    combined_score = 1.0 / (1.0 + nmse)

    return {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "combined_score": float(combined_score),
    }

def evaluate(program_path: str, task_name: str) -> Dict[str, Any]:
    """
    Main function to load and evaluate a scaling law discovery program based on the task name.

    The user's `discover_scaling_law` function signature must strictly match the order of `scaling_vars` in `TASK_CONFIG`.
    For example, for `vocab_scaling_law`, the signature should be:
    def discover_scaling_law(
        train_Non_vocab_parameters, train_vocab_size, train_num_characters, train_lossu,
        test_Non_vocab_parameters, test_vocab_size, test_num_characters
    ) -> Tuple[np.ndarray, Any]:
    """
    # Get task configuration
    config = TASK_CONFIG[task_name]
    scaling_vars_names = config["scaling_vars"]
    response_var_name = config["response_var"]

    try:
        spec = importlib.util.spec_from_file_location("user_program", program_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not create module spec from {program_path}", file=sys.stderr)
            return get_failure_result()

        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "discover_scaling_law"):
            print("Error: Program must have a 'discover_scaling_law' function.", file=sys.stderr)
            return get_failure_result()

        discover_scaling_law_func = program.discover_scaling_law
    except Exception as e:
        print(f"Failed to load or find the required function in the program: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return get_failure_result()

    # Load data
    train_data = load_data(task_name, train=True)
    test_data = load_data(task_name, train=False)

    if not train_data or not test_data:
        print(f"Error: Failed to load training or test data for task '{task_name}'.", file=sys.stderr)
        return get_failure_result()

    # Dynamically aggregate training data
    train_scaling_vars = [np.array([p[var][0] for p in train_data]) for var in scaling_vars_names]
    train_response = np.array([p[response_var_name][0] for p in train_data])

    # Dynamically aggregate test data
    test_scaling_vars = [np.array([p[var][0] for p in test_data]) for var in scaling_vars_names]
    true_test_response = np.array([p[response_var_name][0] for p in test_data])

    # Combine training and test response variables for calculating normalized metrics
    total_response = np.concatenate((train_response, true_test_response))

    try:
        # Use * operator to dynamically pass all variables as parameters
        predicted_response, equation_info = discover_scaling_law_func(
            *train_scaling_vars, train_response,
            *test_scaling_vars
        )
    except Exception as e:
        print(f"An error occurred while running 'discover_scaling_law': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return get_failure_result()

    # Calculate final normalized metrics
    metrics = calculate_final_metrics(predicted_response, true_test_response, total_response)
    if "error" in metrics:
        print(f"Evaluation failed: {metrics['error']}", file=sys.stderr)
        return get_failure_result()

    # Prepare final results
    result = {
        "nmse": metrics["nmse"],
        "nmae": metrics["nmae"],
        "r2": metrics["r2"],
        "combined_score": metrics["combined_score"],
        "equation": equation_info,
    }

    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <task_name> <path_to_your_program.py>")
        print("Available tasks for this evaluator:", ", ".join([t for t, c in TASK_CONFIG.items() if not c['control_vars']]))
        sys.exit(1)

    task_name, program_path = sys.argv[1], sys.argv[2]

    if task_name not in TASK_CONFIG:
        print(f"Error: Unknown task '{task_name}'.", file=sys.stderr)
        sys.exit(1)

    if TASK_CONFIG[task_name]["control_vars"]:
        print(f"Error: Task '{task_name}' requires control variables and is not supported by this evaluator.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(program_path):
        print(f"Error: File not found at '{program_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"# Evaluating program '{program_path}' for task '{task_name}'")
    results = evaluate(program_path, task_name)

    if "error" in results:
        print("\n--- Evaluation Failed ---")
        print(f"Reason: {results.get('equation', 'Unknown error')}")
        print(f"Details: {results.get('error', 'No additional details.')}")
        print("-------------------------\n")
    else:
        print("\n--- Evaluation Results ---")
        print(f"Equation Discovered: {results['equation']}")
        print(f"Normalized MSE (NMSE): {results['nmse']:.6f}")
        print(f"Normalized MAE (NMAE): {results['nmae']:.6f}")
        print(f"R-squared (R2):      {results['r2']:.6f}")
        print(f"Combined Score:        {results['combined_score']:.6f}")
        print("------------------------\n")