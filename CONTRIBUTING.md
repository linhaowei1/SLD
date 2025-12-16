# Contributing to SLDBench

Thank you for your interest in contributing to SLDBench! This guide will walk you through the process of adding new scaling law discovery tasks to the benchmark.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Prepare Your Data](#1-prepare-your-data)
  - [2. Upload Data to Hugging Face Hub](#2-upload-data-to-hugging-face-hub)
  - [3. Register Your Task Schema](#3-register-your-task-schema)
  - [4. Create a Configuration File](#4-create-a-configuration-file)
  - [5. Test Your Task Locally](#5-test-your-task-locally)
  - [6. Submit a Pull Request](#6-submit-a-pull-request)
- [Data Format Specification](#data-format-specification)
- [Configuration File Reference](#configuration-file-reference)
- [Best Practices](#best-practices)
- [Example: Adding a New Task](#example-adding-a-new-task)
- [Getting Help](#getting-help)

---

## Overview

SLDBench is designed to be extensible. Each task in the benchmark represents a unique scaling law discovery challenge, characterized by:

- **Input features**: Variables that influence the scaling behavior (e.g., model size, data size, learning rate)
- **Target variable**: The quantity to predict (e.g., loss, performance metric)
- **Training data**: Experiments used to fit the scaling law parameters
- **Test data**: Held-out experiments used to evaluate extrapolation accuracy

---

## Prerequisites

Before contributing a new task, ensure you have:

1. **Experimental data** from LLM training/evaluation runs with clear input-output relationships
2. A **Hugging Face account** for uploading datasets
3. **Local development environment** set up (see [README.md](README.md#installation))
4. Understanding of the **scaling phenomenon** you want to model

---

## Step-by-Step Guide

### 1. Prepare Your Data

Your data must be organized into two splits:

| Split | Purpose | Typical Size |
|:------|:--------|:-------------|
| `train` | Parameter fitting during evolution | 50-5000 data points |
| `test` | Extrapolation evaluation | 20-1000 data points |

**Important considerations:**

- **Train/Test Split Strategy**: The test split should represent an **extrapolation regime** (e.g., larger models, more data, different hyperparameter ranges) rather than random sampling. This tests the agent's ability to discover generalizable laws.

- **Group Column**: All data must include a `group` column. Groups allow the benchmark to handle multiple related sub-tasks (e.g., different model families, datasets, or experimental conditions). If your task doesn't have natural groups, use a single group name like `"default"`.

- **Data Quality**: Ensure your data is:
  - Free of duplicates
  - Properly normalized/scaled if needed
  - Representative of the scaling phenomenon

**Example data structure:**

```
train.csv:
group,feature_1,feature_2,feature_3,target
model_A,1e8,1e9,0.001,2.5
model_A,1e9,1e10,0.001,2.1
model_B,1e8,1e9,0.001,2.8
...

test.csv:
group,feature_1,feature_2,feature_3,target
model_A,1e10,1e11,0.001,1.8  # Extrapolation to larger scale
model_B,1e10,1e11,0.001,2.2
...
```

### 2. Upload Data to Hugging Face Hub

SLDBench uses the Hugging Face Hub as its central data repository. Follow these steps:

#### 2.1 Create a Dataset Repository

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login
```

#### 2.2 Prepare Your Dataset Files

Create a directory structure for your task:

```
your_task_name/
├── train.parquet  # or .csv, .json
└── test.parquet
```

#### 2.3 Upload to Hub

**Option A: Upload to the official SLDBench repository (recommended)**

Contact the maintainers at [linhaowei@pku.edu.cn](mailto:linhaowei@pku.edu.cn) to have your data added to the official `pkuHaowei/sldbench` repository. This ensures consistency and discoverability.

**Option B: Create your own repository (for testing)**

```python
from datasets import Dataset, DatasetDict
import pandas as pd

# Load your data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Create Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Push to Hub
dataset_dict.push_to_hub(
    "your-username/your-task-name",
    private=False  # Set to True for testing
)
```

### 3. Register Your Task Schema

Add your task to `TASK_SCHEMA_MAP` in `data_loader.py`:

```python
TASK_SCHEMA_MAP = {
    # ... existing tasks ...
    
    "your_task_name": {
        "feature_names": ["feature_1", "feature_2", "feature_3"],  # Column names for input features
        "target_name": "target",  # Column name for the target variable
    },
}
```

**For multi-dimensional targets** (e.g., predicting loss for multiple domains):

```python
"your_task_name": {
    "feature_names": ["proportion_1", "proportion_2", "proportion_3"],
    "target_name": ["loss_1", "loss_2", "loss_3"],  # List of target columns
},
```

### 4. Create a Configuration File

Create `configs/your_task_name.yaml`:

```yaml
# Configuration for your_task_name discovery with OpenEvolve
max_iterations: 50
checkpoint_interval: 1
log_level: "INFO"
random_seed: 42

# LLM configuration
llm:
  api_base: "https://api.openai.com/v1"  # Or your preferred endpoint
  max_tokens: 16384
  timeout: 240
  retries: 10
  retry_delay: 10

# Prompt configuration - CRITICAL: Customize this for your task
prompt:
  system_message: |
    You are an expert in scaling laws and machine learning who specializes in discovering 
    and improving scaling law functions for different LLM training scenarios. Your task is 
    to evolve both the `scaling_law_func` function and the `fit_scaling_law` optimization 
    algorithm to better model the relationship between [DESCRIBE YOUR VARIABLES].

    **IMPORTANT: The scaling law function must use no more than [N] parameters.**

    Focus on mathematical accuracy, cross-dataset generalization, parameter efficiency, 
    and numerical stability.

    **DATA CHARACTERISTICS:**
    - Features: [feature_1, feature_2, feature_3] - [DIMENSIONALITY]D input
    - Labels: [target] - [scalar/vector] output
    - Dataset size: [TRAIN_SIZE] training points, [TEST_SIZE] test points
    - Feature ranges:
      - feature_1: [MIN] to [MAX] ([DESCRIPTION])
      - feature_2: [MIN] to [MAX] ([DESCRIPTION])
      - feature_3: [MIN] to [MAX] ([DESCRIPTION])
    - Target range: [MIN] to [MAX]
    - Key observations: [DESCRIBE KNOWN PATTERNS OR BEHAVIORS]

    The function signatures must remain:

    ```python
    def scaling_law_func(data_points, params):
        # data_points: (N, [NUM_FEATURES]) array with columns [feature_1, feature_2, ...]
        # params: Array of up to [MAX_PARAMS] parameters
        # Returns: Predicted [target] values

    def fit_scaling_law(data_points, loss_values):
        # data_points: (N, [NUM_FEATURES]) array with columns [feature_1, feature_2, ...]
        # loss_values: Array of corresponding [target] values
        # Returns: Optimized parameters (up to [MAX_PARAMS] parameters)
    ```

    Write all improvements between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers.

    You are not allowed to use input-dependent features in scaling_law_func, e.g., median / min / max / etc.

  num_top_programs: 3
  num_diverse_programs: 2
  use_template_stochasticity: true

# Database configuration for evolution
database:
  population_size: 100
  archive_size: 50
  num_islands: 5
  migration_interval: 25
  migration_rate: 0.1
  elite_selection_ratio: 0.1
  exploration_ratio: 0.2
  exploitation_ratio: 0.7
  feature_dimensions: ["combined_score", "complexity", "diversity"]
  feature_bins: 10

# Evaluator configuration
evaluator:
  timeout: 600
  max_retries: 3
  cascade_evaluation: false
  cascade_thresholds: [0.3, 0.6]
  parallel_evaluations: 4
  use_llm_feedback: false

# Evolution settings
diff_based_evolution: false
max_code_length: 100000
```

### 5. Test Your Task Locally

#### 5.1 Verify Data Loading

```bash
# Test that your task loads correctly
python -c "
from data_loader import load_data

# Test training data
train_data = load_data('your_task_name', train=True)
print(f'Training groups: {list(train_data.keys())}')
for key, (X, y) in train_data.items():
    print(f'  {key}: X={X.shape}, y={y.shape}')

# Test test data
test_data = load_data('your_task_name', train=False)
print(f'Test groups: {list(test_data.keys())}')
for key, (X, y) in test_data.items():
    print(f'  {key}: X={X.shape}, y={y.shape}')
"
```

#### 5.2 Run a Quick Evolution Test

```bash
# Run with reduced iterations for testing
EVAL_TASK_NAME="your_task_name" \
uv run openevolve-run \
  --config configs/your_task_name.yaml \
  init_program.py evaluator.py \
  --output results/your_task_name/test_run
```

#### 5.3 Evaluate a Discovered Program

```bash
EVAL_TASK_NAME="your_task_name" \
uv run python evaluator.py \
  results/your_task_name/test_run/best/best_program.py
```

### 6. Submit a Pull Request

Once your task is working locally:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b add-your-task-name`
3. **Commit your changes**:
   - `data_loader.py` (schema registration)
   - `configs/your_task_name.yaml`
   - Any documentation updates
4. **Push to your fork**: `git push origin add-your-task-name`
5. **Open a Pull Request** with:
   - Description of the scaling phenomenon
   - Data source and methodology
   - Expected baseline performance (if known)
   - Any relevant references/papers

---

## Data Format Specification

### Required Columns

| Column | Type | Description |
|:-------|:-----|:------------|
| `group` | string | Identifier for data grouping (e.g., model family, dataset) |
| Feature columns | float | Input variables for the scaling law |
| Target column(s) | float | Output variable(s) to predict |

### Supported File Formats

- **Parquet** (recommended for large datasets)
- **CSV**
- **JSON**

### Example: Single-Target Task

```json
[
  {"group": "gpt", "params": 1e8, "data": 1e9, "loss": 2.5},
  {"group": "gpt", "params": 1e9, "data": 1e10, "loss": 2.1},
  {"group": "llama", "params": 1e8, "data": 1e9, "loss": 2.6}
]
```

### Example: Multi-Target Task

```json
[
  {"group": "exp1", "mix_a": 0.5, "mix_b": 0.5, "loss_a": 2.1, "loss_b": 2.3},
  {"group": "exp2", "mix_a": 0.7, "mix_b": 0.3, "loss_a": 2.0, "loss_b": 2.5}
]
```

---

## Configuration File Reference

### Root-Level Parameters

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `max_iterations` | int | Number of evolution generations | 50 |
| `checkpoint_interval` | int | Save checkpoint every N iterations | 1 |
| `log_level` | string | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | INFO |
| `random_seed` | int | Random seed for reproducibility | 42 |

### LLM Configuration (`llm`)

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `api_base` | string | API endpoint URL | OpenAI |
| `max_tokens` | int | Maximum tokens per request | 16384 |
| `timeout` | int | Request timeout in seconds | 240 |
| `retries` | int | Number of retry attempts | 10 |
| `retry_delay` | int | Delay between retries in seconds | 10 |

### Prompt Configuration (`prompt`)

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `system_message` | string | System prompt describing the task |
| `num_top_programs` | int | Number of top programs to include in prompt |
| `num_diverse_programs` | int | Number of diverse programs for exploration |
| `use_template_stochasticity` | bool | Enable stochastic prompt variations |

### Database Configuration (`database`)

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `population_size` | int | Population size per generation | 100 |
| `archive_size` | int | Archive size for elite programs | 50 |
| `num_islands` | int | Number of islands for island model | 5 |
| `migration_interval` | int | Generations between migrations | 25 |
| `migration_rate` | float | Fraction of population migrating | 0.1 |
| `elite_selection_ratio` | float | Top percentage considered elite | 0.1 |
| `exploration_ratio` | float | Exploration weight | 0.2 |
| `exploitation_ratio` | float | Exploitation weight | 0.7 |

### Evaluator Configuration (`evaluator`)

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `timeout` | int | Evaluation timeout in seconds | 600 |
| `max_retries` | int | Maximum evaluation retries | 3 |
| `parallel_evaluations` | int | Number of concurrent evaluations | 4 |
| `cascade_evaluation` | bool | Enable cascading evaluation | false |
| `use_llm_feedback` | bool | Use LLM for evaluation feedback | false |

---

## Best Practices

### Data Quality

1. **Ensure sufficient data**: Aim for at least 50 training points and 20 test points
2. **Cover diverse regimes**: Training data should span a meaningful range of input values
3. **Design meaningful extrapolation**: Test data should probe regimes beyond training (larger models, more data, etc.)
4. **Remove outliers**: Clean data of obvious measurement errors

### Prompt Engineering

1. **Be specific about data characteristics**: Include feature ranges, scales, and known patterns
2. **Specify parameter constraints**: Limit the number of parameters to prevent overfitting
3. **Describe the phenomenon**: Help the agent understand what scaling behavior to expect
4. **Include relevant domain knowledge**: Reference known scaling laws or theoretical expectations

### Task Design

1. **Make it challenging but feasible**: The task should require discovery, not just curve fitting
2. **Avoid trivial solutions**: Ensure simple power laws don't achieve perfect scores
3. **Test extrapolation**: The benchmark evaluates generalization, not interpolation
4. **Document thoroughly**: Explain the scientific context and practical relevance

---

## Example: Adding a New Task

Let's walk through adding a hypothetical "Context Length Scaling Law" task:

### Step 1: Prepare Data

```python
import pandas as pd
import numpy as np

# Simulated experimental data
np.random.seed(42)

# Training data: models up to 7B, context up to 8K
train_data = []
for model_size in [1e8, 5e8, 1e9, 3e9, 7e9]:
    for context_len in [512, 1024, 2048, 4096, 8192]:
        loss = 3.0 * (model_size ** -0.1) * np.log(context_len) / 10 + np.random.normal(0, 0.05)
        train_data.append({
            "group": "transformer",
            "model_params": model_size,
            "context_length": context_len,
            "loss": loss
        })

# Test data: extrapolate to 13B+ models, 16K+ context
test_data = []
for model_size in [13e9, 30e9, 70e9]:
    for context_len in [16384, 32768, 65536]:
        loss = 3.0 * (model_size ** -0.1) * np.log(context_len) / 10 + np.random.normal(0, 0.05)
        test_data.append({
            "group": "transformer",
            "model_params": model_size,
            "context_length": context_len,
            "loss": loss
        })

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
```

### Step 2: Register Schema

In `data_loader.py`:

```python
TASK_SCHEMA_MAP = {
    # ... existing tasks ...
    "context_length_scaling_law": {
        "feature_names": ["model_params", "context_length"],
        "target_name": "loss",
    },
}
```

### Step 3: Create Config

Create `configs/context_length_scaling_law.yaml`:

```yaml
max_iterations: 50
checkpoint_interval: 1
log_level: "INFO"
random_seed: 42

llm:
  api_base: "https://api.openai.com/v1"
  max_tokens: 16384
  timeout: 240
  retries: 10
  retry_delay: 10

prompt:
  system_message: |
    You are an expert in scaling laws for large language models. Your task is to discover 
    how language modeling loss scales with model size and context length.

    **IMPORTANT: The scaling law function must use no more than 5 parameters.**

    **DATA CHARACTERISTICS:**
    - Features: [model_params, context_length] - 2D input
    - Labels: loss - scalar output
    - Training: 25 points (models 100M-7B, context 512-8K)
    - Test: 9 points (models 13B-70B, context 16K-64K) - EXTRAPOLATION
    - Model parameter range: 1e8 to 7e10
    - Context length range: 512 to 65536
    - Loss range: approximately 1.5 to 3.5
    
    Key observation: Loss appears to decrease with model size and increase logarithmically 
    with context length.

    The function signatures must remain:

    ```python
    def scaling_law_func(data_points, params):
        # data_points: (N, 2) array with columns [model_params, context_length]
        # params: Array of up to 5 parameters
        # Returns: Predicted loss values

    def fit_scaling_law(data_points, loss_values):
        # data_points: (N, 2) array
        # loss_values: Array of loss values
        # Returns: Optimized parameters
    ```

    Write all improvements between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers.

  num_top_programs: 3
  num_diverse_programs: 2
  use_template_stochasticity: true

database:
  population_size: 100
  archive_size: 50
  num_islands: 5
  migration_interval: 25
  migration_rate: 0.1
  elite_selection_ratio: 0.1
  exploration_ratio: 0.2
  exploitation_ratio: 0.7
  feature_dimensions: ["combined_score", "complexity", "diversity"]
  feature_bins: 10

evaluator:
  timeout: 600
  max_retries: 3
  cascade_evaluation: false
  cascade_thresholds: [0.3, 0.6]
  parallel_evaluations: 4
  use_llm_feedback: false

diff_based_evolution: false
max_code_length: 100000
```

### Step 4: Test and Validate

```bash
# Verify data loading
python -c "from data_loader import load_data; print(load_data('context_length_scaling_law'))"

# Run evolution
EVAL_TASK_NAME="context_length_scaling_law" \
uv run openevolve-run \
  --config configs/context_length_scaling_law.yaml \
  init_program.py evaluator.py \
  --output results/context_length_scaling_law/run_1
```

---

## Getting Help

- **Questions**: Open an issue on GitHub
- **Data hosting**: Contact [linhaowei@pku.edu.cn](mailto:linhaowei@pku.edu.cn) to add your data to the official repository
- **Collaboration**: We welcome research collaborations on scaling law discovery

---

## License

By contributing to SLDBench, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to SLDBench! Your work helps advance our understanding of how AI models scale.

