#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

## --- Configuration ---

# Usage: ./run_benchmark.sh [jobs_per_model]
# Default is 4 to be safe.
JOBS_PER_MODEL=${1:-4}

# OPTIONAL: Safety cap for the TOTAL number of jobs across all models on the machine.
# Set to empty string "" to disable global limiting (unlimited total jobs).
GLOBAL_MAX_JOBS=64

# Array of task configurations to run
tasks=(
    "sft_scaling_law"
    "data_constrained_scaling_law"
    "moe_scaling_law"
    "vocab_scaling_law"
    "domain_mixture_scaling_law"
    "lr_bsz_scaling_law"
    "parallel_scaling_law"
    "easy_question_scaling_law"
)

# Array of models to test
models=(
    "gpt-5"
    "claude-sonnet-4-5-20250929"
    "claude-haiku-4-5-20251001"
    "gemini-2.5-flash"
    "gemini-3-pro-preview"
    "o4-mini"
)

RESULTS_BASE_DIR="./results"

## --- Graceful Shutdown ---

# Kill the entire process group (script + all subshells + all python jobs)
cleanup() {
    echo -e "\n🚨 Caught Signal. Killing all descendant processes..."
    # 'kill 0' sends the signal to every process in the current process group.
    # We suppress the self-kill error message.
    kill 0 2>/dev/null
    exit 1
}

# Trap INT (Ctrl+C) and TERM
trap cleanup INT TERM EXIT

## --- Helper Function ---

# Check global concurrency across the entire machine (for the current user)
wait_for_global_slots() {
    if [ -z "$GLOBAL_MAX_JOBS" ]; then return; fi
    
    # Count python/openevolve jobs belonging to this user
    # Note: This is a rough heuristic to prevent machine melting
    while true; do
        current_jobs=$(pgrep -c -u "$(whoami)" -f "python|openevolve")
        if [ "$current_jobs" -lt "$GLOBAL_MAX_JOBS" ]; then
            break
        fi
        # Sleep randomly to prevent race conditions between subshells
        sleep $(( ( RANDOM % 5 )  + 1 ))
    done
}

## --- Core Logic ---

run_single_job() {
    local task_name=$1
    local model=$2
    local run=$3
    local run_id="run_${run}"
    local output_dir="${RESULTS_BASE_DIR}/${task_name}/${model}/${run_id}"
    local best_program_path="${output_dir}/best/best_program.py"
    local best_eval_log_path="${output_dir}/best_eval.log"

    # Enforce global limit if set
    wait_for_global_slots

    if [ -s "$best_eval_log_path" ]; then
        return
    fi

    # Create a unique log file for this specific job execution to avoid console clutter
    local job_log="${output_dir}/execution.log"
    mkdir -p "$output_dir"

    # echo "Starting ${task_name}/${model}/${run}" >> "$job_log"

    if [ ! -f "$best_program_path" ]; then
        EVAL_TASK_NAME="$task_name" uv run openevolve-run \
            --config "configs/${task_name}.yaml" \
            init_program.py evaluator.py \
            --primary-model "$model" \
            --output "$output_dir" \
            >> "$job_log" 2>&1
    fi

    if [ -f "$best_program_path" ]; then
        EVAL_TASK_NAME="$task_name" uv run python evaluator.py \
            "$best_program_path" \
            > "$best_eval_log_path" 2>&1
    fi
}

## --- Job Orchestration ---

echo "Starting benchmark."
echo "Config: $JOBS_PER_MODEL jobs per model."
[ -n "$GLOBAL_MAX_JOBS" ] && echo "Global Cap: Max $GLOBAL_MAX_JOBS concurrent python processes on machine."

total_runs_per_config=5

for model in "${models[@]}"; do
    (
        echo "--- 🚀 Launching pool for [$model] ---"
        for task in "${tasks[@]}"; do
            for run in $(seq 1 $total_runs_per_config); do
                
                # Per-Model Limit Check
                # 'jobs -r' counts only running jobs in this subshell
                while [[ $(jobs -r -p | wc -l) -ge $JOBS_PER_MODEL ]]; do
                    # Wait for *any* job in this subshell to finish
                    wait -n
                done

                # Run in background
                run_single_job "$task" "$model" "$run" &
            done
        done
        wait
        echo "--- ✅ Model [$model] finished all tasks ---"
    ) &
done

# Main script waits for all model subshells
wait
echo "✅ All tasks completed!"