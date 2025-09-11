# Daily LLM

## Summary

This repository provides an HPC/SLURM job submission framework for running Large Language Models (LLMs) on GPU-enabled computing clusters. The project offers scripts to run various open-source LLMs from the [Hugging Face Hub](https://huggingface.co/models) using SLURM job scheduling, with support for both single job execution and batch processing through job arrays, automated email notifications, and comprehensive resource management for academic and research computing environments.

## Key Components

### 1. Python Scripts - Model Loading and Inference

- Scripts named `<model_name>.py` provide basic model loading using HuggingFace transformers
- `gemma29b-loop.py`: Continuous query processing system that monitors a `query/` directory for .txt files and generates responses in `answer/` directory

### 2. SLURM Shell Scripts - Job Submission Templates

- Scripts named after the Python scripts with the `.sh` file extension for single job execution
- Array versions (`-array.sh`) for batch processing with job arrays (1-10 instances)
- Configured for A100 GPUs (40GB or 80GB variants)
- Email notifications on job completion
- Resource allocation: 4 CPUs, 30GB RAM, 72-hour time limit
- **Parameterized execution**: `run_model.sh` accepts optional job name and Python script parameters

## Features

- Automated job submission with email notifications
- Support for both preemptible and general GPU partitions
- Job output logging to `outputs/` directory
- Execution time tracking and reporting
- Query-response pipeline for batch processing LLM requests

## Usage Example

To run inference with the generic model runner script:

```bash
# Run with default settings (EleutherAI/gpt-neox-20b model)
python run_model.py

# Run with a different model
python run_model.py --model "meta-llama/Meta-Llama-3-8B"

# Run with custom query and token limit
python run_model.py --query "What is quantum computing?" --max-tokens 2000

# Run only pipeline inference
python run_model.py --pipeline-only

# Run only direct model inference
python run_model.py --direct-only
```

### SLURM Job Submission

The parameterized batch script allows flexible job submission:

```bash
# Submit with default settings (job name: runmodel, script: run_model.py)
sbatch run_model.sh

# Submit with custom job name
sbatch run_model.sh experiment1

# Submit with custom job name and Python script
sbatch run_model.sh myexp custom_model.py
```

## Roadmap

1. Parameterising batch scripts to make them (hopefully) a "one size fits all"
2. Placing scripts in a separate scripts folder
3. Creating a logs folder for results