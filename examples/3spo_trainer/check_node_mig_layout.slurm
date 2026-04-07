#!/bin/bash
#SBATCH --job-name=check_mig_layout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:10
#SBATCH --mem=8G
#SBATCH --partition=H200
#SBATCH --output=logs/%j_check_mig_layout.out
#SBATCH --error=logs/%j_check_mig_layout.err

set -euo pipefail

echo "=== Node Info ==="
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo

echo "=== nvidia-smi -L ==="
nvidia-smi -L
echo

echo "=== MIG Summary Per Physical GPU ==="
nvidia-smi -L | awk '
    /^GPU [0-9]+:/ {
        gpu=$2
        sub(":", "", gpu)
        gpu_header[gpu]=$0
        gpu_order[++gpu_count]=gpu
        next
    }
    /MIG-[^)]+/ {
        if (gpu != "") {
            has_mig[gpu]=1
            mig_count[gpu]++
            if (mig_lines[gpu] == "") {
                mig_lines[gpu]=$0
            } else {
                mig_lines[gpu]=mig_lines[gpu] "\n" $0
            }
        }
    }
    END {
        total_with_mig=0
        total_without_mig=0
        total_mig_instances=0

        for (i=1; i<=gpu_count; ++i) {
            g=gpu_order[i]
            print gpu_header[g]
            if (has_mig[g]) {
                printf "  STATUS: has MIG (%d instance(s))\n", mig_count[g]
                print mig_lines[g]
                total_with_mig++
                total_mig_instances += mig_count[g]
            } else {
                print "  STATUS: no MIG"
                total_without_mig++
            }
            print ""
        }

        print "=== Totals ==="
        printf "PHYSICAL_GPUS=%d\n", gpu_count
        printf "GPUS_WITH_MIG=%d\n", total_with_mig
        printf "GPUS_WITHOUT_MIG=%d\n", total_without_mig
        printf "TOTAL_MIG_INSTANCES=%d\n", total_mig_instances
    }
'
