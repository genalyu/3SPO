#!/bin/bash
#SBATCH --job-name=alfworld_grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1g.35gb:2
#SBATCH --mem=360G
#SBATCH --partition=H200
#SBATCH --output=logs/%j_alfworld_grpo.out
#SBATCH --error=logs/%j_alfworld_grpo.err

# 1. Environment Setup
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-alf
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline  # Enable offline WandB
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
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
# 增加这个变量，告诉 NCCL 忽略由于 P2P 被禁用引发的额外拓扑检查
export NCCL_IGNORE_DISABLED_P2P=1
# 允许 vLLM 忽略 P2P 检查，在 MIG 环境下这是必要的
export VLLM_SKIP_P2P_CHECK=1
# 如果仍然报错 Duplicate GPU，尝试开启 Ray 的 MIG 支持
export RAY_EXPERIMENTAL_NO_GCS_CHECK=1
# 强制导出环境变量给 Ray Actor (关键！)
export RAY_EXPORT_CUDA_VISIBLE_DEVICES=1
# 禁止 Ray 在 actor 启动时自动清空/重写 CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# 某些集群需要强制禁用 IB 通信来配合 MIG，如果报错继续，请取消下面这行的注释
export NCCL_IB_DISABLE=1

# 开启 NCCL 调试日志，如果出错，日志里会告诉你具体是哪两张卡在打架
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# 原生 MIG GRES 模式：
# 调度层申请 n 个 MIG，但训练时只使用来自 2 张不同物理 GPU 的 2 个 MIG。
REQUESTED_MIGS=2
REQUIRED_DISTINCT_GPUS=2
TRAIN_GPUS_PER_NODE=2
BAD_MIG_EXIT_CODE=42

# 调试：打印环境中的 GPU 变量
echo "DEBUG: SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "DEBUG: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L


ALL_GPU_LISTING=$(nvidia-smi -L)
DISTINCT_PARENT_GPUS=""
SELECTED_MIGS=""

# 有些集群在 MIG GRES 模式下会把 CUDA_VISIBLE_DEVICES 设为索引而不是 MIG UUID。
# 这里统一从当前可见的 nvidia-smi -L 中挑选 MIG，并优先在不同物理 GPU 间分散。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty under MIG GRES allocation."
    exit "$BAD_MIG_EXIT_CODE"
fi

if echo "$CUDA_VISIBLE_DEVICES" | grep -q 'MIG-'; then
    CANDIDATE_MIGS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n')
else
    echo "DEBUG: CUDA_VISIBLE_DEVICES contains indices, deriving MIG candidates from nvidia-smi -L"
    CANDIDATE_MIGS=$(echo "$ALL_GPU_LISTING" | grep -o 'MIG-[^)]*' || true)
fi

ASSIGNED_MIG_COUNT=$(echo "$CANDIDATE_MIGS" | sed '/^\s*$/d' | wc -l | tr -d ' ')
if [ "$ASSIGNED_MIG_COUNT" -lt "$REQUESTED_MIGS" ]; then
    echo "ERROR: Visible MIG candidates only $ASSIGNED_MIG_COUNT, require at least $REQUESTED_MIGS."
    echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    exit "$BAD_MIG_EXIT_CODE"
fi

while IFS= read -r mig_uuid; do
    mig_uuid=$(echo "$mig_uuid" | xargs)
    [ -z "$mig_uuid" ] && continue

    parent_gpu=$(echo "$ALL_GPU_LISTING" | awk -v target="$mig_uuid" '
        /^GPU [0-9]+:/ {
            current_gpu=$2
            sub(":", "", current_gpu)
        }
        index($0, target) {
            print current_gpu
            exit
        }
    ')

    if [ -z "$parent_gpu" ]; then
        echo "ERROR: Could not map MIG $mig_uuid to a physical GPU."
        exit "$BAD_MIG_EXIT_CODE"
    fi

    if ! echo ",$DISTINCT_PARENT_GPUS," | grep -q ",$parent_gpu,"; then
        if [ -n "$DISTINCT_PARENT_GPUS" ]; then
            DISTINCT_PARENT_GPUS="${DISTINCT_PARENT_GPUS},${parent_gpu}"
            SELECTED_MIGS="${SELECTED_MIGS},${mig_uuid}"
        else
            DISTINCT_PARENT_GPUS="$parent_gpu"
            SELECTED_MIGS="$mig_uuid"
        fi
    fi
done <<< "$CANDIDATE_MIGS"

DISTINCT_PARENT_COUNT=$(echo "$DISTINCT_PARENT_GPUS" | awk -F',' '{print NF}')
if [ "$DISTINCT_PARENT_COUNT" -lt "$REQUIRED_DISTINCT_GPUS" ]; then
    echo "ERROR: Assigned MIGs are not spread across $REQUIRED_DISTINCT_GPUS physical GPUs."
    echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "ERROR: Distinct parent GPUs=$DISTINCT_PARENT_GPUS"
    exit "$BAD_MIG_EXIT_CODE"
fi

REAL_MIG_UUIDS=$(echo "$SELECTED_MIGS" | awk -F',' -v required="$TRAIN_GPUS_PER_NODE" '
    {
        for (i = 1; i <= NF && i <= required; ++i) {
            printf "%s%s", $i, (i < required ? "," : "")
        }
    }
')
export CUDA_VISIBLE_DEVICES=$REAL_MIG_UUIDS
export VERL_ALL_MIG_IDS=$REAL_MIG_UUIDS
echo "DEBUG: Distinct parent GPUs=$DISTINCT_PARENT_GPUS"
echo "DEBUG: Detected MIG UUIDs: $REAL_MIG_UUIDS"
echo "DEBUG: Final CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

ROLLOUT_TP_SIZE=2
echo "DEBUG: REQUESTED_MIGS=$REQUESTED_MIGS"
echo "DEBUG: REQUIRED_DISTINCT_GPUS=$REQUIRED_DISTINCT_GPUS"
echo "DEBUG: TRAIN_GPUS_PER_NODE=$TRAIN_GPUS_PER_NODE"
echo "DEBUG: ROLLOUT_TP_SIZE=$ROLLOUT_TP_SIZE"
# --- MIG 环境关键修复结束 ---
# 确保旧的 Ray 进程不会带着旧变量残留
ray stop --force

ENGINE=${1:-vllm}
num_cpus_per_env_worker=0.1

train_data_size=8
val_data_size=8
group_size=8
mode="mean_std_norm"

model_path="/public/home/genalyu/models/Qwen2.5-1.5B-Instruct"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=32 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=$TRAIN_GPUS_PER_NODE \
    trainer.default_local_dir=$HOME/project/verl-agent/checkpoints/grpo_qwen2.5_1.5b_alf \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.total_epochs=400 \
    trainer.val_before_train=True $@ 2>&1 | tee logs/grpo_qwen2.5_1.5b.log
