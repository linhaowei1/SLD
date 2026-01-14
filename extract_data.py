#!/usr/bin/env python3
"""
Script to extract data from sldagent_260113 results and create JSONL files.
"""

import os
import json
import re
from pathlib import Path

SOURCE_DIR = "/WillDevExt/linhaoweiShare/code/SLD/results/sldagent-v1"
TARGET_DIR = "/WillDevExt/linhaoweiShare/code/scaling_law_discovery_results/data"

SCALING_LAWS = [
    "easy_question_scaling_law",
    "parallel_scaling_law",
    "lr_bsz_scaling_law",
    "domain_mixture_scaling_law",
    "vocab_scaling_law",
    "moe_scaling_law",
    "data_constrained_scaling_law",
    "sft_scaling_law",
]

AGENT_NAME = "SLDAgent"


def extract_r2_from_eval_log(eval_log_path):
    """Extract R-squared value from best_eval.log file."""
    try:
        with open(eval_log_path, 'r') as f:
            content = f.read()
        # Look for R-squared line like: "  R-squared (R²):        0.847918" or negative values
        match = re.search(r'R-squared.*?:\s*(-?[\d.]+)', content)
        if match:
            r2 = float(match.group(1))
            # Clip very negative values to (-1, 1) range
            r2 = max(-1.0, min(1.0, r2))
            return r2
    except Exception as e:
        print(f"Error reading {eval_log_path}: {e}")
    return None


def read_best_program(best_program_path):
    """Read the best_program.py file content."""
    try:
        with open(best_program_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {best_program_path}: {e}")
    return None


def extract_data_for_scaling_law(scaling_law_name):
    """Extract data for a single scaling law from all models and runs."""
    source_path = Path(SOURCE_DIR) / scaling_law_name
    if not source_path.exists():
        print(f"Warning: {source_path} does not exist")
        return []

    records = []
    
    # Iterate over model directories
    for model_dir in sorted(source_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Iterate over run directories
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                continue
            
            # Check for best_eval.log
            eval_log_path = run_dir / "best_eval.log"
            # Check for best_program.py (in best/ subdirectory)
            best_program_path = run_dir / "best" / "best_program.py"
            
            if not eval_log_path.exists():
                print(f"Warning: {eval_log_path} does not exist")
                continue
            
            if not best_program_path.exists():
                print(f"Warning: {best_program_path} does not exist")
                continue
            
            # Extract R2 value
            r2 = extract_r2_from_eval_log(eval_log_path)
            if r2 is None:
                print(f"Warning: Could not extract R2 from {eval_log_path}")
                continue
            
            # Read best program
            solution = read_best_program(best_program_path)
            if solution is None:
                print(f"Warning: Could not read {best_program_path}")
                continue
            
            # Create record
            record = {
                "model_name": model_name,
                "reward_r2": r2,
                "solution": solution,
                "agent_name": AGENT_NAME,
                "task": scaling_law_name
            }
            records.append(record)
            print(f"  Extracted: {model_name}/{run_dir.name} - R2={r2:.6f}")
    
    return records


def main():
    """Main function to extract all data."""
    for scaling_law in SCALING_LAWS:
        print(f"\nProcessing {scaling_law}...")
        
        # Extract records
        records = extract_data_for_scaling_law(scaling_law)
        
        if not records:
            print(f"  No records extracted for {scaling_law}")
            continue
        
        # Read existing JSONL file if it exists
        target_file = Path(TARGET_DIR) / f"{scaling_law}.jsonl"
        existing_records = []
        if target_file.exists():
            with open(target_file, 'r') as f:
                for line in f:
                    if line.strip():
                        existing_records.append(json.loads(line))
        
        # Filter out any existing SLDAgent records to avoid duplicates
        existing_records = [r for r in existing_records if r.get('agent_name') != AGENT_NAME]
        
        # Combine existing and new records
        all_records = existing_records + records
        
        # Write to JSONL file
        with open(target_file, 'w') as f:
            for record in all_records:
                f.write(json.dumps(record) + '\n')
        
        print(f"  Wrote {len(records)} new records to {target_file}")
        print(f"  Total records in file: {len(all_records)}")


if __name__ == "__main__":
    main()

