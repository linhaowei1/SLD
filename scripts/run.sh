#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

## --- Configuration ---

# Set the maximum number of parallel jobs PER MODEL.
# Usage: ./run_benchmark.sh 8
# If no argument is given, it defaults to 4.
PARALLELISM_DEGREE=${1:-4}

# Array of task configurations to run
tasks=(
    "sft_scaling_law"
    "data_constrained_scaling_law"
    "moe_scaling_law"
    "vocab_scaling_law"
    "domain_mixture_scaling_law"
    "lr_bsz_scaling_law"
    "parallel_scaling_law"
)

# Array of models to test
models=(
    "claude-sonnet-4-20250514"
)

# Base directory for results
RESULTS_BASE_DIR="./results"

## --- Graceful Shutdown ---

# This function is called when the script receives a signal to terminate (e.g., Ctrl+C).
cleanup() {
    echo -e "\n🚨 Caught Ctrl+C. Terminating all background jobs..."
    # Get the process IDs of all background jobs started by this script.
    pids_to_kill=$(jobs -p)
    if [ -n "$pids_to_kill" ]; then
        # Terminate all the jobs. The '2>/dev/null' suppresses errors
        # if a job has already finished.
        kill $pids_to_kill 2>/dev/null
    fi
    echo "All background jobs terminated. Exiting now."
    exit 1
}

# Set the trap. The 'cleanup' function will be called when the script
# receives an INT (interrupt, from Ctrl+C) or TERM signal.
trap cleanup INT TERM

## --- Core Logic ---

# This function encapsulates a single unit of work: one evolution and its evaluation.
run_single_job() {
    local task_name=$1
    local model=$2
    local run=$3
    local run_id="run_${run}"
    local output_dir="${RESULTS_BASE_DIR}/${task_name}/${model}/${run_id}"
    local best_program_path="${output_dir}/best/best_program.py"
    local best_eval_log_path="${output_dir}/best_eval.log"

    # --- Skip Condition ---
    # A job is complete only if its log file exists AND is not empty.
    # The '-s' check handles both missing and empty files.
    if [ -s "$best_eval_log_path" ]; then
        echo "--- Skipping: [Task: $task_name] [Model: $model] [Run: $run] (valid result exists) ---"
        return
    fi

    echo "--- Processing: [Task: $task_name] [Model: $model] [Run: $run] ---"

    # Ensure the output directory exists for subsequent steps.
    mkdir -p "$output_dir"

    # --- Evolution Step ---
    # Run the full evolution process only if the best program does not exist.
    if [ ! -f "$best_program_path" ]; then
        echo "   -> No best program found. Evolving for ${task_name}/${model}/${run_id}..."
        EVAL_TASK_NAME="$task_name" uv run openevolve-run \
            --config "configs/${task_name}.yaml" \
            init_program.py evaluator.py \
            --primary-model "$model" \
            --secondary-model "$model" \
            --output "$output_dir"
    else
        echo "   -> Best program already exists. Skipping evolution."
    fi

    # --- Evaluation Step ---
    # After a potential evolution, check for the program again. If it exists,
    # run (or re-run) the evaluation.
    if [ -f "$best_program_path" ]; then
        echo "   -> Evaluating best program for ${task_name}/${model}/${run_id}..."
        EVAL_TASK_NAME="$task_name" uv run python evaluator.py \
            "$best_program_path" \
            > "$best_eval_log_path"
    else
        # This case occurs if the evolution step was required but failed to produce a program.
        echo "   -> WARNING: Evolution failed. No best program found for ${task_name}/${model}/${run_id}. Evaluation skipped."
    fi

    echo "--- Finished: [Task: $task_name] [Model: $model] [Run: $run] ---"
}

## --- Job Orchestration ---

echo "Starting benchmark with a parallelism degree of $PARALLELISM_DEGREE PER MODEL."
# Calculate and display the total number of jobs that will be executed
total_runs_per_config=3
total_jobs=$((${#tasks[@]} * ${#models[@]} * total_runs_per_config))
echo "Total jobs to run: $total_jobs"

# Loop through each model and start a dedicated, parallelized process for it.
for model in "${models[@]}"; do
    # Run the task processing for each model in a separate subshell `(...)`
    # and launch that subshell in the background `&`.
    # This ensures each model has its own independent job pool.
    (
        echo "--- Launching tasks for model [$model] with inner parallelism of [$PARALLELISM_DEGREE] ---"

        # Loop through all tasks and runs for the current model
        for task in "${tasks[@]}"; do
            for run in $(seq 1 $total_runs_per_config); do
                # Check the number of jobs running *within this subshell*.
                # This enforces the per-model parallelism limit.
                while [[ $(jobs -p | wc -l) -ge $PARALLELISM_DEGREE ]]; do
                    # Wait for any job *within this subshell* to complete.
                    wait -n
                done

                # Launch the actual job in the background.
                run_single_job "$task" "$model" "$run" &
            done
        done
        
        # After launching all jobs for this model, wait for them to finish
        # before the subshell exits.
        wait
        echo "--- ✅ All tasks for model [$model] are complete. ---"
    ) &
done

# Final wait at the main script level to ensure all backgrounded model-processing
# subshells have completed.
echo "All model task groups launched. Waiting for all models to complete..."
wait

echo "✅ All tasks completed!"