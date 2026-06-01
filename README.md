# 3SPO: State-Score-Supervised Policy Optimization

[![Paper](https://img.shields.io/badge/arXiv-3SPO-blue)](https://arxiv.org/abs/XXXX.XXXXX)

## Requirements

- Python 3.10+
- PyTorch 2.4+
- [verl](https://github.com/volcengine/verl)
- vLLM or SGLang as the rollout engine
- Ray

## Quick Start

### 1. Data Preparation

```bash
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128
```

> **Note:** The dataset `hiyouga/geometry3k` is only used as a modality/size indicator — the actual task data comes from environment interactions during training. See [prepare.py](examples/data_preprocess/prepare.py) for details.

### 2. Run 3SPO Training

#### Local (single node)

```bash
# Qwen2.5-1.5B on ALFWorld (2 GPUs)
bash examples/3spo_trainer/run_alfworld_1.5B.sh

# Qwen2.5-7B on ALFWorld (8 GPUs)
bash examples/3spo_trainer/run_alfworld_7B.sh

# GRPO baseline (Qwen2.5-1.5B, 2 GPUs)
bash examples/3spo_trainer/run_alfworld_grpo.sh
```

#### SLURM (multi-GPU / MIG)

```bash
# 3SPO
sbatch examples/3spo_trainer/run_alfworld.slurm

# GRPO baseline
sbatch examples/3spo_trainer/run_alfworld_grpo.slurm
```

### 3. Evaluate

Evaluation uses the HGPO recipe (`recipe.hgpo.main_hgpo`) with `adv_estimator=hgpo` for checkpoint evaluation:

```bash
bash examples/3spo_trainer/run_qwen2.5_1.5b_alfworld_eval.sh
# or
sbatch examples/3spo_trainer/run_qwen2.5_1.5b_alfworld_eval.slurm
```

> **Note:** Edit `CHECKPOINTS_DIR` and `eval_experiment_names` in the eval script to point to your trained checkpoint before running.

### Rollout Engine

```bash
# Use vLLM (default)
bash examples/3spo_trainer/run_alfworld_1.5B.sh vllm

# Use SGLang
bash examples/3spo_trainer/run_alfworld_1.5B.sh sglang
```

## MIG Environment Notes

When running on NVIDIA MIG-partitioned GPUs, the SLURM scripts (`*.slurm`) handle:

- MIG UUID detection and GPU assignment
- `CUDA_VISIBLE_DEVICES` override per Ray worker
- Disabling P2P communication (`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`)

For non-MIG setups, use the `*.sh` scripts directly.

## Merge LoRA Checkpoints

```bash
python3 examples/3spo_trainer/model_merger.py \
    --model-path <path-to-checkpoint> \
    --target-dir <output-dir>
```
