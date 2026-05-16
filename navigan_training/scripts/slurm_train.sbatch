#!/bin/bash
# UCT HPC Slurm template for multi-GPU NaviGAN training.
#
# Submit:   sbatch scripts/slurm_train.sbatch
# Monitor:  squeue -u $USER ; tail -f logs/navigan-<JOBID>.out
#
# Edit the placeholders marked TODO before submitting.

#SBATCH --job-name=navigan
#SBATCH --output=logs/navigan-%j.out
#SBATCH --error=logs/navigan-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4               # 4 GPUs on one node
#SBATCH --time=24:00:00
#SBATCH --mem=32G
# TODO: set your cluster's partition + account
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>

set -euo pipefail

mkdir -p logs

# TODO: source your environment (conda env, module load, virtualenv, ...)
# Examples — pick one:
# module load python/3.10 cuda/12.1
# source ~/envs/navigan/bin/activate
# conda activate navigan

REPO_ROOT="$(pwd)"
CONFIG="${1:-configs/zara1_scratch.yaml}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[slurm] Job ${SLURM_JOB_ID}, node $(hostname), ${NUM_GPUS} GPU(s), config=${CONFIG}"

# Pick a free master port for torchrun's rendezvous
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

cd "${REPO_ROOT}"
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    --distributed \
    --config "${CONFIG}"
