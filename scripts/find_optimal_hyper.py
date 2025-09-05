#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finds and evaluates all 'law.py' files within a directory structure against a
baseline model on a test dataset, using a best@N metric.

This script performs three main tasks:
1.  Discovery: Scans a base directory for all 'law.py' files matching the
    expected structure (e.g., .../task/model/run_*/best/law.py).
2.  Batch Evaluation: For each law, it computes the achieved loss and MAE 
    for the best@N predicted hyperparameters, comparing it against a 'StepFun' 
    baseline for N in [1, 2, 4, 8].
3.  Aggregation: It compiles all results and presents two summary tables:
    a) Performance of the single best law for each model.
    b) Average performance across all laws for each model.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


def load_law_module(law_path: Path) -> ModuleType:
    """Dynamically imports the 'law.py' file as a module."""
    if not law_path.exists():
        raise FileNotFoundError(f"Law file not found at '{law_path}'")
    
    # Use a unique module name to avoid conflicts in the loop
    module_name = f"law_module_{law_path.stem}_{law_path.parent.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, law_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {law_path}")
        
    law_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = law_module
    spec.loader.exec_module(law_module)
    return law_module


def predict_stepfun_params(N: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicts optimal lr and bsz using the baseline 'stepfun' formulas."""
    lr_pred = 1.79 * (N ** -0.713) * (D ** 0.307)
    bs_pred = (0.58 * (D ** 0.571)) / 2048
    return lr_pred, bs_pred


def evaluate_single_law(law_module: ModuleType, df: pd.DataFrame, top_n_values: list[int]) -> dict:
    """
    Evaluates the best@N performance of a single law.py module and the stepfun law.
    Returns a dictionary of raw loss and MAE values for each N.
    """
    # 1. Find ground truth optimal hyperparameters for each group
    group_cols = ['data_size', 'non_embedding_param_size']
    optimal_df = df.loc[df.groupby(group_cols)['lm_loss'].idxmin()].reset_index(drop=True)
    if optimal_df.empty:
        print("Warning: No valid groups found in the test set for evaluation.", file=sys.stderr)
        nan_results = {}
        for n in top_n_values:
            nan_results.update({
                f'law_loss_at_{n}': float('nan'), f'stepfun_loss_at_{n}': float('nan'),
                f'law_mae_at_{n}': float('nan'), f'stepfun_mae_at_{n}': float('nan')
            })
        return nan_results

    loss_optimal_true = optimal_df['lm_loss'].values
    
    # Dictionaries to store the best@N losses for this run, keyed by N
    pred_loss_law = {n: [] for n in top_n_values}
    target_loss_stepfun = {n: [] for n in top_n_values}

    # Get stepfun predictions for all optimal groups at once
    D_optimal = optimal_df['data_size'].values
    N_optimal = optimal_df['non_embedding_param_size'].values
    lr_pred_stepfun, bs_pred_stepfun = predict_stepfun_params(N_optimal, D_optimal)
    
    group_name = list(law_module.FITTED_PARAMS.keys())[0]

    # Loop over each group defined by the optimal points
    for i, _ in optimal_df.iterrows():
        d_val = D_optimal[i]
        n_val = N_optimal[i]
        group_rows = df[(df['data_size'] == d_val) & (df['non_embedding_param_size'] == n_val)]
        
        actual_losses_in_group = group_rows['lm_loss'].values

        # --- `law.py` evaluation for this group ---
        input_data = [row.to_dict() for _, row in group_rows[law_module.FEATURE_NAMES].iterrows()]
        predictions = law_module.law(input_data, group=group_name)
        pred_losses_grid = np.array([p[law_module.TARGET_NAMES[0]] for p in predictions])
        
        sorted_pred_indices = np.argsort(pred_losses_grid)
        
        for n in top_n_values:
            top_n_indices = sorted_pred_indices[:n]
            best_actual_loss = np.min(actual_losses_in_group[top_n_indices])
            pred_loss_law[n].append(best_actual_loss)

        # --- StepFun evaluation for this group ---
        lr_pred, bs_pred = lr_pred_stepfun[i], bs_pred_stepfun[i]
        exp_lrs, exp_bss = group_rows['lr'].values, group_rows['bsz'].values
        
        log_dist_sq = (np.log(exp_lrs) - np.log(lr_pred))**2 + (np.log(exp_bss) - np.log(bs_pred))**2
        sorted_dist_indices = np.argsort(log_dist_sq)

        for n in top_n_values:
            closest_n_indices = sorted_dist_indices[:n]
            best_actual_loss = np.min(actual_losses_in_group[closest_n_indices])
            target_loss_stepfun[n].append(best_actual_loss)

    # Calculate final metrics for each N
    results = {}
    for n in top_n_values:
        # Raw average loss
        results[f'law_loss_at_{n}'] = np.mean(pred_loss_law[n])
        results[f'stepfun_loss_at_{n}'] = np.mean(target_loss_stepfun[n])
        # MAE (for improvement calculation and finding the best law)
        results[f'law_mae_at_{n}'] = mean_absolute_error(loss_optimal_true, np.array(pred_loss_law[n]))
        results[f'stepfun_mae_at_{n}'] = mean_absolute_error(loss_optimal_true, np.array(target_loss_stepfun[n]))
        
    return results


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Find and evaluate all 'law.py' files in a directory using best@N.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/WillDevExt/linhaoweiShare/evosld-results/lr_bsz_scaling_law",
        help="The base directory containing the experiment results."
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        nargs='+',
        default=[1],
        help="List of N values for best@N evaluation (e.g., -n 1 2 4 8)."
    )
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    top_n_values = sorted(list(set(args.top_n)))
    
    law_files = sorted(list(base_path.glob('**/best/law.py')))
    
    if not law_files:
        print(f"Error: No '**/best/law.py' files found under '{base_path}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(law_files)} 'law.py' files to evaluate for best@{top_n_values}.")

    print("Loading test dataset 'pkuHaowei/sldbench:lr_bsz_scaling_law'...")
    dataset = load_dataset('pkuHaowei/sldbench', 'lr_bsz_scaling_law', split='test')
    df_test = dataset.to_pandas()
    if 'bs' in df_test.columns:
        df_test = df_test.rename(columns={'bs': 'bsz'})

    all_results = []
    for law_path in tqdm(law_files, desc="Processing all law files"):
        try:
            parts = law_path.parts
            record = {
                'task': parts[-5], 'model': parts[-4], 'run_id': parts[-3],
            }
            law_module = load_law_module(law_path)
            results_dict = evaluate_single_law(law_module, df_test, top_n_values)
            record.update(results_dict)
            all_results.append(record)

        except Exception as e:
            print(f"\n❗️ Failed to process {law_path}: {e}", file=sys.stderr)

    if not all_results:
        print("No results were successfully collected.", file=sys.stderr)
        sys.exit(1)
        
    results_df = pd.DataFrame(all_results)
    
    # --- Table 1: Best Law Performance ---
    best_runs_df = results_df.loc[results_df.groupby('model')['law_mae_at_1'].idxmin()]
    
    best_law_summary_data = []
    for _, row in best_runs_df.iterrows():
        record = {'model': row['model'], 'run_id': row['run_id']}
        for n in top_n_values:
            record[f'Law Loss @{n}'] = row[f'law_loss_at_{n}']
            record[f'StepFun Loss @{n}'] = row[f'stepfun_loss_at_{n}']
            law_mae = row[f'law_mae_at_{n}']
            stepfun_mae = row[f'stepfun_mae_at_{n}']
            improvement = (stepfun_mae - law_mae) / stepfun_mae * 100 if stepfun_mae != 0 else 0
            record[f'Improv. % @{n}'] = improvement
        best_law_summary_data.append(record)

    best_law_summary_df = pd.DataFrame(best_law_summary_data)
    
    print("\n\n" + "="*80)
    print(f"====== Best Law Performance for Task: {Path(args.base_dir).name} ======")
    print("="*80)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.max_rows', None, 'display.width', 1000):
        print(best_law_summary_df)

    # --- Table 2: Average Performance Across All Runs ---
    agg_dict = {}
    for n in top_n_values:
        agg_dict[f'Law Loss @{n}'] = (f'law_loss_at_{n}', 'mean')
        agg_dict[f'StepFun Loss @{n}'] = (f'stepfun_loss_at_{n}', 'mean')
        # Also aggregate MAE to calculate average improvement
        agg_dict[f'law_mae_at_{n}'] = (f'law_mae_at_{n}', 'mean')
        agg_dict[f'stepfun_mae_at_{n}'] = (f'stepfun_mae_at_{n}', 'mean')

    summary = results_df.groupby('model').agg(**agg_dict)
    
    # Calculate average improvement and drop intermediate MAE columns
    for n in top_n_values:
        law_mae_mean = summary[f'law_mae_at_{n}']
        stepfun_mae_mean = summary[f'stepfun_mae_at_{n}']
        summary[f'Improv. % @{n}'] = (stepfun_mae_mean - law_mae_mean) / stepfun_mae_mean * 100
        summary = summary.drop(columns=[f'law_mae_at_{n}', f'stepfun_mae_at_{n}'])

    summary = summary.reset_index()

    print("\n\n" + "="*80)
    print(f"====== Average Performance (across all runs) for Task: {Path(args.base_dir).name} ======")
    print("="*80)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.max_rows', None, 'display.width', 1000):
        print(summary)
        
    print(f"\n✅ Finished processing {len(all_results)}/{len(law_files)} law files.")

if __name__ == "__main__":
    main()