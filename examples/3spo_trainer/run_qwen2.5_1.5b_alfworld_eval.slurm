#!/bin/bash
#SBATCH --job-name=alfworld_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1g.35gb:2
#SBATCH --mem=512G
#SBATCH --partition=H200
#SBATCH --output=logs/%j_alfworld_eval.out
#SBATCH --error=logs/%j_alfworld_eval.err

# 1. Environment Setup
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-alf
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_DIR=${WANDB_DIR}
export HF_ENDPOINT=https://hf-mirror.com
# 在 NVIDIA 环境下，显式清除 AMD 和 NPU 特有的变量，防止后端冲突报错
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset ASCEND_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 彻底禁用导致 "Duplicate GPU" 报错的 P2P 通信和共享内存
# 因为 MIG 实例之间在物理上无法直接 P2P 访问
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_SKIP_P2P_CHECK=1
export RAY_EXPERIMENTAL_NO_GCS_CHECK=1
export RAY_EXPORT_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_IB_DISABLE=1

# 开启 NCCL 调试日志
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# MIG 环境配置
TRAIN_GPUS_PER_NODE=2

# 调试：打印环境中的 GPU 变量
echo "DEBUG: SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "DEBUG: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

ALL_GPU_LISTING=$(nvidia-smi -L)

# 收集所有可见的 MIG UUID（用于 worker 隔离和 VERL_ALL_MIG_IDS）
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "$CUDA_VISIBLE_DEVICES" | grep -q 'MIG-'; then
    REAL_MIG_UUIDS="$CUDA_VISIBLE_DEVICES"
elif [ -n "${SLURM_JOB_GPUS:-}" ] && echo "$SLURM_JOB_GPUS" | grep -q 'MIG-'; then
    REAL_MIG_UUIDS="$SLURM_JOB_GPUS"
else
    # 从 nvidia-smi 提取 MIG UUID
    REAL_MIG_UUIDS=$(echo "$ALL_GPU_LISTING" | grep -oP 'MIG-[^)]+' | head -n $TRAIN_GPUS_PER_NODE | tr '\n' ',' | sed 's/,$//')
fi

# Ray 不识别 MIG UUID，CUDA_VISIBLE_DEVICES 必须用数字索引
# 保留 UUID 在 VERL_ALL_MIG_IDS 中供 worker 隔离使用
TRAIN_GPU_INDICES=""
for i in $(seq 0 $((TRAIN_GPUS_PER_NODE - 1))); do
    if [ -n "$TRAIN_GPU_INDICES" ]; then
        TRAIN_GPU_INDICES="${TRAIN_GPU_INDICES},${i}"
    else
        TRAIN_GPU_INDICES="${i}"
    fi
done
export CUDA_VISIBLE_DEVICES=$TRAIN_GPU_INDICES
export VERL_ALL_MIG_IDS=$REAL_MIG_UUIDS
echo "DEBUG: Detected MIG UUIDs: $REAL_MIG_UUIDS"
echo "DEBUG: Final CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

ENGINE=${1:-vllm}
num_cpus_per_env_worker=0.1
eval_out=True
seeds=(123)

eval_experiment_names=(
    "3spo_qwen2.5_1.5b_alfworld_eval_True"
)
CHECKPOINTS_DIR=~/project/verl-agent/checkpoints/alfworld/1833
train_data_size=8
val_data_size=8
group_size=8
mode="mean_std_norm"

# 确保旧的 Ray 进程不会带着旧变量残留
ray stop --force

# loop: first by experiment name, then by seed
for eval_experiment_name in "${eval_experiment_names[@]}"; do
    history_length=$(echo "$eval_experiment_name" | sed -n 's/.*k\([0-9]\+\).*/\1/p')
    history_length=${history_length:-2}
    echo "history_length: $history_length"

    eval_dir="${CHECKPOINTS_DIR}"
    if [ ! -d "$eval_dir" ]; then
        echo "Error: checkpoint directory does not exist: $eval_dir, skip."
        continue
    fi

    log_dir="logs"
    mkdir -p "$log_dir"

    for seed in "${seeds[@]}"; do
        echo "=========================================="
        echo "Running experiment: $eval_experiment_name, seed: $seed"
        echo "=========================================="

        temp_log="${log_dir}/output_seed${seed}.log"

        python3 -m recipe.hgpo.main_hgpo \
            algorithm.adv_estimator=hgpo \
            data.train_files=$HOME/data/verl-agent/text/train.parquet \
            data.val_files=$HOME/data/verl-agent/text/test.parquet \
            data.train_batch_size=$train_data_size \
            data.val_batch_size=$val_data_size \
            data.max_prompt_length=4096 \
            data.max_response_length=512 \
            data.filter_overlong_prompts=True \
            data.truncation='left' \
            data.return_raw_chat=True \
            actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=256 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.enable_chunked_prefill=False \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.free_cache_engine=False \
            actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.use_invalid_action_penalty=True \
            actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
            algorithm.use_kl_in_reward=False \
            algorithm.gamma=0.95 \
            algorithm.hgpo.mode=$mode \
            env.env_name=alfworld/AlfredTWEnv \
            env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
            env.seed=$seed \
            env.history_length=$history_length \
            env.max_steps=50 \
            env.rollout.n=$group_size \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name="qwen2.5_1.5b_alfworld_eval" \
            trainer.experiment_name="${eval_experiment_name}_seed${seed}_eval_out${eval_out}" \
            trainer.n_gpus_per_node=$TRAIN_GPUS_PER_NODE \
            trainer.nnodes=1 \
            trainer.save_freq=-1 \
            trainer.test_freq=5 \
            trainer.total_epochs=160 \
            trainer.default_local_dir=$eval_dir \
            trainer.val_only=True \
            trainer.val_out=$eval_out \
            trainer.val_before_train=True $@ 2>&1 | tee "$temp_log"
        echo "Completed: $eval_experiment_name, seed: $seed"
        echo ""
    done
done
