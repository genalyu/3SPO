#!/bin/bash
#SBATCH --job-name=3spo_webshop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --partition=H200
#SBATCH --output=logs/%j_webshop.out
#SBATCH --error=logs/%j_webshop.err

# 1. Environment Setup
set -x
ENGINE=${1:-vllm}
ulimit -u 65536
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline  # Enable offline WandB
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# unset CUDA_VISIBLE_DEVICES
# unset HIP_VISIBLE_DEVICES
# unset ROCR_VISIBLE_DEVICES  # 重点清除这一项

# 允许 vLLM 忽略 P2P 检查，在 MIG 环境下这是必要的
export VLLM_SKIP_P2P_CHECK=1
# 如果仍然报错 Duplicate GPU，尝试开启 Ray 的 MIG 支持
export RAY_EXPERIMENTAL_NO_GCS_CHECK=1
# 强制禁用导致 MIG 环境下 NCCL 报错的 IB/P2P/SHM 通信
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

# 确保旧的 Ray 进程不会带着旧变量残留
ray stop --force
num_cpus_per_env_worker=0.1
HOME=/public/home/genalyu
train_data_size=16
val_data_size=128
group_size=8
mode="mean_norm"

model_path="/public/home/genalyu/models/Qwen2.5-1.5B-Instruct"
# 3SPO Parameters
alpha=1.0
xi=10
zeta=0.1
beta=5.0
omega_k=0.1

# 2. Data Preparation
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $((val_data_size * 2))

# 3. Launch 3SPO Training
python3 -m gigpo.main_3spo \
    algorithm.adv_estimator=grpo \
    +algorithm.spo3_alpha=$alpha \
    +algorithm.spo3_xi=$xi \
    +algorithm.spo3_zeta=$zeta \
    +algorithm.spo3_beta=$beta \
    +algorithm.spo3_omega_k=$omega_k \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='3spo_qwen2.5_1.5b_slurm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/project/verl-agent/checkpoints/webshop/$SLURM_JOB_ID \
    trainer.save_freq=100 \
    trainer.test_freq=200 \
    trainer.total_epochs=400 \
    trainer.val_before_train=True
