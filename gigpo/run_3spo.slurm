#!/bin/bash
#SBATCH --job-name=3spo_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# 1. Environment Setup
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

# 2. Distributed Training Config
n_gpus=8
train_batch_size=64
group_size=8

# 3. 3SPO Hyperparameters
alpha=1.0    # For lambda(t) annealing
xi=10        # Max failures threshold
zeta=0.1     # Min success rate threshold
beta=5.0     # Sigmoid scale for adaptive rollout

# 4. Data Preprocessing (Example for AlfWorld)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_batch_size \
    --val_data_size 128

# 5. Launch 3SPO Training
# We use the custom main_3spo.py which integrates the 3SPO logic
python3 -m gigpo.main_3spo \
    algorithm.adv_estimator=grpo \
    algorithm.spo3_alpha=$alpha \
    algorithm.spo3_xi=$xi \
    algorithm.spo3_zeta=$zeta \
    algorithm.spo3_beta=$beta \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.train_batch_size=$train_batch_size \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    env.env_name=alfworld/AlfredTWEnv \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    trainer.project_name='3spo_agent' \
    trainer.experiment_name='3spo_qwen2.5_7b_step_grpo' \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.total_epochs=100
