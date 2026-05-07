#!/bin/bash
# Run case study on ALFWorld with trained model
# Usage: bash scripts/run_case_study.sh [model_path] [num_tasks]

set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-alf

MODEL_PATH=${1:-"~/project/verl-agent/checkpoints/alfworld/1833/actor/huggingface"}
NUM_TASKS=${2:-7}
MAX_STEPS=${3:-20}
TEMPERATURE=${4:-0.4}

python3 scripts/case_study_alfworld.py \
    --model-path "$MODEL_PATH" \
    --num-tasks "$NUM_TASKS" \
    --max-steps "$MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --output-dir "case_study_output" \
    --dtype "bfloat16"
