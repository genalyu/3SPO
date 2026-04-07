#!/bin/bash
#SBATCH --job-name=3spo_alfworld
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1g.35gb:4
#SBATCH --mem=512G
#SBATCH --partition=H200
#SBATCH --output=logs/%j_alfworld.out
#SBATCH --error=logs/%j_alfworld.err

# 1. Environment Setup
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-alf
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline  # Enable offline WandB
export HF_ENDPOINT=https://hf-mirror.com
# 在 NVIDIA 环境下，显式清除 AMD 和 NPU 特有的变量，防止后端冲突报错
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES 
unset ASCEND_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3
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

# 原生 MIG GRES 模式：严格要求 4 个 MIG，且必须来自 4 张不同物理 GPU。
REQUIRED_MIGS=4

# 调试：打印环境中的 GPU 变量
echo "DEBUG: SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "DEBUG: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Slurm 在 MIG GRES 模式下应该直接把分配给作业的 MIG 暴露到 CUDA_VISIBLE_DEVICES。
ASSIGNED_MIGS="$CUDA_VISIBLE_DEVICES"
if [ -z "$ASSIGNED_MIGS" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty under MIG GRES allocation."
    echo "INFO: Requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

if ! echo "$ASSIGNED_MIGS" | grep -q 'MIG-'; then
    echo "ERROR: Expected MIG UUIDs in CUDA_VISIBLE_DEVICES, got: $ASSIGNED_MIGS"
    echo "INFO: Requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

ASSIGNED_MIG_COUNT=$(echo "$ASSIGNED_MIGS" | awk -F',' '{print NF}')
if [ "$ASSIGNED_MIG_COUNT" -lt "$REQUIRED_MIGS" ]; then
    echo "ERROR: Assigned only $ASSIGNED_MIG_COUNT MIG(s), require $REQUIRED_MIGS."
    echo "INFO: Requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

ALL_GPU_LISTING=$(nvidia-smi -L)
DISTINCT_PARENT_GPUS=""
SELECTED_MIGS=""
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
        echo "INFO: Requeueing job $SLURM_JOB_ID"
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
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
done <<< "$(echo "$ASSIGNED_MIGS" | tr ',' '\n')"

DISTINCT_PARENT_COUNT=$(echo "$DISTINCT_PARENT_GPUS" | awk -F',' '{print NF}')
if [ "$DISTINCT_PARENT_COUNT" -lt "$REQUIRED_MIGS" ]; then
    echo "ERROR: Assigned MIGs are not spread across $REQUIRED_MIGS physical GPUs."
    echo "ERROR: CUDA_VISIBLE_DEVICES=$ASSIGNED_MIGS"
    echo "ERROR: Distinct parent GPUs=$DISTINCT_PARENT_GPUS"
    echo "INFO: Requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

REAL_MIG_UUIDS=$(echo "$SELECTED_MIGS" | awk -F',' -v required="$REQUIRED_MIGS" '
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

TRAIN_GPUS_PER_NODE=$REQUIRED_MIGS
ROLLOUT_TP_SIZE=2
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
#     --val_data_size $val_data_size

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
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
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
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='3spo_qwen2.5_1.5b_slurm' \
    trainer.n_gpus_per_node=$TRAIN_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/project/verl-agent/checkpoints/alfworld/$SLURM_JOB_ID \
    trainer.save_freq=100 \
    trainer.test_freq=200 \
    trainer.total_epochs=400 \
    trainer.val_before_train=True
