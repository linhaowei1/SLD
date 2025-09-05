import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_log_file(file_path):
    """
    Parses a best_eval.log file to extract the R-squared value.

    Args:
        file_path (str): The path to the log file.

    Returns:
        float or None: The R-squared value if found, otherwise None.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Regex to find the R-squared value in the "Final Test Results" block
            match = re.search(r"R-squared \(R²\):\s+([0-9.]+)", content)
            if match:
                return float(match.group(1))
    except FileNotFoundError:
        # This is expected if a run failed, so we don't print an error.
        return None
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return None
    return None

def main():
    """
    Main function to find log files, extract data, and generate result tables.
    """
    parser = argparse.ArgumentParser(
        description="Extract and summarize benchmark results from log files."
    )
    parser.add_argument(
        "results_dir",
        nargs='?',
        default="/WillDevExt/linhaoweiShare/evosld-results",
        help="The base directory where results are stored. Defaults to the path in the script."
    )
    args = parser.parse_args()

    results_base_dir = args.results_dir
    print(f"🔍 Searching for results in: {results_base_dir}\n")

    if not os.path.isdir(results_base_dir):
        print(f"❌ Error: Base directory not found at '{results_base_dir}'")
        return

    # --- Configuration (from run_benchmark.sh) ---
    tasks = [
        "sft_scaling_law",
        "data_constrained_scaling_law",
        "moe_scaling_law",
        "vocab_scaling_law",
        "domain_mixture_scaling_law",
        "lr_bsz_scaling_law",
        "parallel_scaling_law"
    ]
    models = [
        "o4-mini",
        "gpt-5",
        "gemini-2.5-flash"
    ]
    total_runs = 10

    # --- Data Collection ---
    # A nested dictionary to store results: {task: {model: [r2_scores]}}
    results_data = defaultdict(lambda: defaultdict(list))
    
    found_logs_count = 0
    for task in tasks:
        for model in models:
            for i in range(1, total_runs + 1):
                run_id = f"run_{i}"
                log_path = os.path.join(results_base_dir, task, model, run_id, "best_eval.log")
                
                r2_score = parse_log_file(log_path)
                if r2_score is not None:
                    results_data[task][model].append(r2_score)
                    found_logs_count += 1
    
    print(f"✅ Found and processed {found_logs_count} log files.\n")

    # --- Table Generation ---
    # Initialize DataFrames with tasks as index and models as columns
    best_r2_df = pd.DataFrame(index=tasks, columns=models, dtype=float)
    avg_std_r2_df = pd.DataFrame(index=tasks, columns=models, dtype=str)

    for task in tasks:
        for model in models:
            scores = results_data[task][model]
            if scores:
                # Table 1: Best R-squared
                best_r2_df.loc[task, model] = np.max(scores)
                
                # Table 2: Average ± Standard Deviation
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                avg_std_r2_df.loc[task, model] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                # Handle cases where no logs were found for a combination
                best_r2_df.loc[task, model] = np.nan
                avg_std_r2_df.loc[task, model] = "N/A"

    # --- Display Results ---
    print("--- Table 1: Best R² (Max of 10 Runs) ---")
    print(best_r2_df.to_string(float_format="%.4f"))
    print("\n" + "="*50 + "\n")
    print("--- Table 2: R² (Average ± Std. Dev. of 10 Runs) ---")
    print(avg_std_r2_df)
    print("\n")


if __name__ == "__main__":
    main()
