#!/bin/bash

# Array of task configurations
tasks=(
    # "data_constrained_scaling_law"
    # "domain_mixture_scaling_law"
    # "lr_scaling_law"
    # "moe_scaling_law"
    # "rectified_scaling_law"
    # "vocab_scaling_law"
    "lr_scaling_law"
)

# Create results directory if it doesn't exist
mkdir -p results

# Run each task 3 times
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    
    for run in 1 2 3; do
        run_id="run_${run}"
        echo "  Run $run for $task"
        
        # Create output directory
        mkdir -p "results/${task}/${run_id}"
        
        # Run the evolution
        EVAL_TASK_NAME="$task" uv run openevolve-run \
            --config "configs/${task}.yaml" \
            init_program.py evaluator.py \
            --output "results/${task}/${run_id}"
        
        # Check if best program exists and evaluate
        if [ -f "results/${task}/${run_id}/best/best_program.py" ]; then
            echo "  Evaluating best program for $task run $run"
            EVAL_TASK_NAME="$task" uv run python evaluator.py \
                "results/${task}/${run_id}/best/best_program.py" \
                > "results/${task}/${run_id}/best_eval.log"
        else
            echo "  Warning: No best program found for $task run $run"
        fi
    done
    
    echo "Completed all runs for $task"
    echo ""
done

echo "All tasks completed!"