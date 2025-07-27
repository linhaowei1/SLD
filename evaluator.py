"""
Unified Evaluator for Scaling Law Discovery
-------------------------------------------

This script provides a unified framework for evaluating scaling law discovery programs
across various tasks, guided by the formal problem definition in the project paper.

Key Concepts from the Paper:
- Scaling Variables (x): Inputs to the scaling law function (e.g., tokens, model size).
- Control Variables (c): Define experimental groups; a separate set of parameters (θ)
  is fitted for each unique control setting (e.g., dataset, model architecture).
- Response Variable (y): The target value to be predicted (e.g., loss).

The evaluator operates in two main steps:
1.  FIT: On the training dataset, it fits the parameters of the scaling law for each
    control group, producing a map of {control_group_key: parameters}.
2.  EVALUATE: On the test dataset, it uses the pre-fitted parameters to make
    predictions and calculates normalized metrics (NMSE, NMAE) as specified in the paper.
    
Version 3: Added a --visualize option for the 'moe_scaling_law' task.
"""

import importlib.util
import numpy as np
import pandas as pd
import os
import sys
import time
import traceback
import concurrent.futures
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path for data_loader import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

# --- Task Configuration ---
TASK_CONFIG = {
    "rectified_scaling_law": {
        "scaling_vars": ["data_size"],
        "control_vars": ["model", "dataset"],
        "response_var": "loss_values",
    },
    "domain_mixture_scaling_law": {
        "scaling_vars": ["proportions"],
        "control_vars": ["model_size"],
        "response_var": "loss_values",
    },
    "data_constrained_scaling_law": {
        "scaling_vars": ["data_size", "model_size", "unique_tokens"],
        "control_vars": [],
        "response_var": "loss_values",
    },
    "moe_scaling_law": {
        "scaling_vars": ["num_experts", "total_parameter_count"],
        "control_vars": [],
        "response_var": "loss_values",
    },
    "vocab_scaling_law": {
        "scaling_vars": ["Non_vocab_parameters", "vocab_size", "num_characters"],
        "control_vars": [],
        "response_var": "lossu_values",
    },
}

def get_failure_result() -> Dict[str, float]:
    """Returns a standardized dictionary for failure cases."""
    worst_nmse = 100000.0
    worst_nmae = 100000.0
    worst_r2 = -1.0
    worst_score = 1.0 / (1.0 + worst_nmse)
    
    return {
        "nmse": worst_nmse,
        "nmae": worst_nmae,
        "r2": worst_r2,
        "combined_score": worst_score,
        "error": "Evaluation failed or timed out.",
    }

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=600):
    """Runs a function with a specified timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except (concurrent.futures.TimeoutError, Exception) as e:
            print(f"Function {func.__name__} timed out or failed: {e}", file=sys.stderr)
            raise

def _prepare_data(task_name: str, train: bool) -> Tuple[Dict[Any, Dict[str, np.ndarray]], List[Dict]]:
    """
    Loads and groups data according to the task configuration, handling different
    data structures from the data_loader correctly.
    """
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}")

    config = TASK_CONFIG[task_name]
    data_points = load_data(task_name, train=train)
    if not data_points:
        return {}, []

    grouped_data = {}
    control_vars = config["control_vars"]
    
    # The 'rectified_scaling_law' task is unique: each 'point' is a full group.
    if task_name == "rectified_scaling_law":
        for point in data_points:
            key = tuple(point[cv] for cv in control_vars)
            grouped_data[key] = {
                "scaling_vars": {sv: point[sv] for sv in config["scaling_vars"]},
                "response_var": point[config["response_var"]],
            }
        return grouped_data, data_points

    # Logic for all other tasks that require aggregation.
    temp_groups = {}
    for point in data_points:
        key = "all_data" if not control_vars else tuple(point[cv] for cv in control_vars)
        if key not in temp_groups:
            temp_groups[key] = {
                "scaling_vars": {sv: [] for sv in config["scaling_vars"]},
                "response_var": [],
            }
        
        for sv in config["scaling_vars"]:
            temp_groups[key]["scaling_vars"][sv].append(point[sv])
        temp_groups[key]["response_var"].append(point[config["response_var"]])

    # Convert aggregated lists to final numpy arrays with correct shapes.
    for key, group in temp_groups.items():
        final_group = {"scaling_vars": {}}
        for sv, values in group["scaling_vars"].items():
            # For domain_mixture, stack vectors into a 2D array.
            # For other tasks, flatten into a 1D array.
            if task_name == "domain_mixture_scaling_law":
                 final_group["scaling_vars"][sv] = np.array(values)
            else:
                 final_group["scaling_vars"][sv] = np.array(values).flatten()

        # Same logic for the response variable.
        if task_name == "domain_mixture_scaling_law":
            final_group["response_var"] = np.array(group["response_var"])
        else:
            final_group["response_var"] = np.array(group["response_var"]).flatten()
        
        grouped_data[key] = final_group
        
    return grouped_data, data_points


def calculate_final_metrics(
    predictions: np.ndarray, 
    true_values: np.ndarray, 
    total_response_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculates final evaluation metrics as defined in the paper.
    - NMSE: Normalized by the variance of the total (train + test) dataset.
    - NMAE: Normalized by the mean absolute deviation of the total dataset.
    """
    pred = np.asarray(predictions, dtype=float).flatten()
    true = np.asarray(true_values, dtype=float).flatten()
    total_y = np.asarray(total_response_values, dtype=float).flatten()
    
    valid_mask = ~(np.isnan(pred) | np.isinf(pred))
    if not np.any(valid_mask) or pred.shape != true.shape:
        return {"error": "Prediction failed, is invalid, or has shape mismatch."}
        
    pred_filtered = pred[valid_mask]
    true_filtered = true[valid_mask]

    if len(pred_filtered) == 0:
        return {"error": "No valid data points left after filtering."}

    test_mse = np.mean((true_filtered - pred_filtered) ** 2)
    test_mae = np.mean(np.abs(true_filtered - pred_filtered))
    total_variance = np.var(total_y)
    total_mean_abs_deviation = np.mean(np.abs(total_y - np.mean(total_y)))

    nmse = test_mse / total_variance if total_variance > 0 else test_mse
    nmae = test_mae / total_mean_abs_deviation if total_mean_abs_deviation > 0 else test_mae
    r2 = 1.0 - nmse
    combined_score = 1.0 / (1.0 + nmse)

    return {"nmse": float(nmse), "nmae": float(nmae), "r2": float(r2), "combined_score": float(combined_score)}

def evaluate(
    program_path: str,
    task_name: str,
    use_test_data: bool = False,
    fitted_params_map: Dict[Any, Any] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Main evaluation function, implementing the logic from the paper.
    """
    try:
        spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        fit_scaling_law = program.fit_scaling_law
        scaling_law_func = program.scaling_law_func

        if not use_test_data:
            grouped_data, _ = _prepare_data(task_name, train=True)
            if not grouped_data:
                return {"error": "No training data found."}

            new_fitted_params_map = {}
            for key, group in grouped_data.items():
                scaling_vars_tuple = tuple(group["scaling_vars"].values())
                response_var = group["response_var"]
                params = run_with_timeout(fit_scaling_law, args=(*scaling_vars_tuple, response_var))
                new_fitted_params_map[key] = params
            return {"fitted_params": new_fitted_params_map}
        else:
            if fitted_params_map is None:
                return {"error": "fitted_params_map is required for evaluation."}

            grouped_test_data, _ = _prepare_data(task_name, train=False)
            if not grouped_test_data: return {"error": "No test data found."}

            all_predictions, all_true_values = [], []
            for key, group in grouped_test_data.items():
                if key not in fitted_params_map:
                    print(f"Warning: No params for test group {key}. Skipping.", file=sys.stderr)
                    continue
                params = fitted_params_map[key]
                scaling_vars_tuple = tuple(group["scaling_vars"].values())
                predictions = run_with_timeout(scaling_law_func, args=(*scaling_vars_tuple, params))
                all_predictions.append(np.asarray(predictions).flatten())
                all_true_values.append(group["response_var"])

            if not all_predictions: return get_failure_result()

            final_predictions = np.concatenate(all_predictions)
            final_true_values = np.concatenate(all_true_values)

            _, train_points = _prepare_data(task_name, train=True)
            _, test_points = _prepare_data(task_name, train=False)
            response_var_name = TASK_CONFIG[task_name]["response_var"]
            
            # Use itertools to efficiently flatten the list of arrays/scalars
            all_points = train_points + test_points
            total_response_vals = list(itertools.chain.from_iterable(p[response_var_name] for p in all_points if isinstance(p[response_var_name], (list, np.ndarray))))
            total_response_vals.extend([p[response_var_name] for p in all_points if not isinstance(p[response_var_name], (list, np.ndarray))])


            return calculate_final_metrics(final_predictions, final_true_values, total_response_vals)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result()


def visualize_moe_scaling_law(scaling_law_func: callable, params: Any, save_dir: str = "vis"):
    """
    Generates and saves plots for the MoE scaling law based on fitted parameters.
    """
    print("\n--- Generating Visualization for MoE Scaling Law ---")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "moe_scaling_law_visualization.pdf")

        # Load both training and testing data for a comprehensive plot
        train_data = load_data('moe_scaling_law', train=True)
        test_data = load_data('moe_scaling_law', train=False)
        data = train_data + test_data
        
        if not data:
            print("No data found for visualization.", file=sys.stderr)
            return

        # Extract data into numpy arrays
        loss_values = np.array([d['loss_values'][0] for d in data])
        num_experts = np.array([d['num_experts'][0] for d in data])
        total_parameter_count = np.array([d['total_parameter_count'][0] for d in data])

        # Get unique values for experts and parameters for plotting
        nexp = np.unique(num_experts)
        nparam = np.unique(total_parameter_count)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Loss vs. Total Parameter Count (for fixed numbers of experts)
        param_range = np.geomspace(nparam.min(), nparam.max(), 100)
        for e in sorted(nexp):
            loss_pred = scaling_law_func(np.full_like(param_range, e), param_range, params)
            line, = ax1.plot(param_range, loss_pred, label=f'E = {int(e)}')
            mask = (num_experts == e)
            ax1.scatter(total_parameter_count[mask], loss_values[mask], color=line.get_color(), zorder=5)

        ax1.set_title('Loss vs. Parameters (Fixed Experts)', fontsize=14)
        ax1.set_xlabel('Total Parameter Count', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_xscale('log')
        ax1.grid(True, which="both", ls="--")
        ax1.legend()

        # Plot 2: Loss vs. Number of Experts (for fixed parameter counts)
        expert_range = np.geomspace(nexp.min(), nexp.max(), 100)
        for p in sorted(nparam):
            loss_pred = scaling_law_func(expert_range, np.full_like(expert_range, p), params)
            line, = ax2.plot(expert_range, loss_pred, label=f'P = {p/1e9:.4f}B')
            mask = (total_parameter_count == p)
            ax2.scatter(num_experts[mask], loss_values[mask], color=line.get_color(), zorder=5)

        ax2.set_title('Loss vs. Experts (Fixed Parameters)', fontsize=14)
        ax2.set_xlabel('Number of Experts', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_xscale('log')
        ax2.grid(True, which="both", ls="--")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Visualization saved to: {save_path}")

    except Exception as e:
        print(f"❌ Failed to generate visualization: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Evaluator for Scaling Law Discovery.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("task_name", type=str, choices=list(TASK_CONFIG.keys()),
                        help="The name of the task to evaluate.")
    parser.add_argument("program_path", type=str,
                        help="The path to the Python script defining the scaling law.")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save a visualization plot.\n(Currently only supported for 'moe_scaling_law')")
    
    args = parser.parse_args()
    task_name = args.task_name
    program_path = args.program_path

    if not os.path.exists(program_path):
        print(f"Error: Path '{program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running Evaluation for Task: {task_name} ---")

    print("\n# Step 1: Fitting scaling law on TRAINING data...")
    fit_result = evaluate(program_path, task_name, use_test_data=False)

    if "fitted_params" not in fit_result:
        print("Error: Failed to fit parameters on training data.", file=sys.stderr)
        print(fit_result.get("error", "Unknown fitting error."), file=sys.stderr)
        sys.exit(1)

    fitted_params_map = fit_result["fitted_params"]
    print(f"Successfully fitted parameters for {len(fitted_params_map)} control group(s).")
    
    print("\n# Step 2: Evaluating fitted law on TEST data...")
    test_result = evaluate(
        program_path,
        task_name,
        use_test_data=True,
        fitted_params_map=fitted_params_map,
    )

    if "error" in test_result:
        print("Error: Failed to evaluate on test data.", file=sys.stderr)
        print(test_result["error"], file=sys.stderr)
        sys.exit(1)

    print("\n--- Final Test Results ---")
    print(f"  Normalized MSE (NMSE): {test_result['nmse']:.6f}")
    print(f"  Normalized MAE (NMAE): {test_result['nmae']:.6f}")
    print(f"  R-squared (R2):        {test_result['r2']:.6f}")
    print(f"  Combined Score:        {test_result['combined_score']:.6f}")
    print(f"    Fitted params: {fitted_params_map}")
    print("--------------------------")

    # --- Visualization Step (if requested) ---
    if args.visualize:
        if task_name == "moe_scaling_law":
            try:
                spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
                program = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(program)
                scaling_law_func = program.scaling_law_func
                
                # For moe_scaling_law, there are no control vars, so the key is 'all_data'.
                params_key = "all_data"
                if params_key in fitted_params_map:
                    moe_params = fitted_params_map[params_key]
                    visualize_moe_scaling_law(scaling_law_func, moe_params, program_path.replace('.py', '.png'))
                else:
                    print(f"Could not find fitted parameters under the key '{params_key}' for visualization.", file=sys.stderr)

            except Exception as e:
                print(f"An error occurred during visualization setup: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"\nNote: Visualization is only implemented for 'moe_scaling_law', not for '{task_name}'.")