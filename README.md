# 3SPO: Three-Step Preference Optimization

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
    --train_data_size 8 \
    --val_data_size 8
```

### 2. Run 3SPO Training

#### Local (single node)

```bash
# Qwen2.5-1.5B on ALFWorld
bash examples/3spo_trainer/run_alfworld_1.5B.sh

# Qwen2.5-7B on ALFWorld
bash examples/3spo_trainer/run_alfworld_7B.sh

# GRPO baseline
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

```bash
bash examples/3spo_trainer/run_qwen2.5_1.5b_alfworld_eval.sh
# or
sbatch examples/3spo_trainer/run_qwen2.5_1.5b_alfworld_eval.slurm
```

## Configuration

### Key 3SPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.spo3_alpha` | 1.0 | |
| `algorithm.spo3_xi` | 10 | |
| `algorithm.spo3_zeta` | 0.1 | |
| `algorithm.spo3_beta` | 5.0 | |
| `algorithm.spo3_omega_k` | 0.1 | |

### Common Training Parameters

| Parameter | 1.5B | 7B |
|-----------|------|-----|
| `data.train_batch_size` | 16 | 8 |
| `data.max_prompt_length` | 2048 | 2048 |
| `data.max_response_length` | 512 | 512 |
| `actor.ppo_mini_batch_size` | 256 | 256 |
| `actor.ppo_micro_batch_size_per_gpu` | 32 | 8 |
| `rollout.tensor_model_parallel_size` | 2 | 2 |
| `trainer.n_gpus_per_node` | 2 | 4 |
| `actor.fsdp_config.param_offload` | False | True |
| `actor.fsdp_config.optimizer_offload` | False | True |

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
