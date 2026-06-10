#!/bin/bash
# SPDX-License-Identifier: MIT
# 2-node x 4-GPU MI300A (gfx942) eval of the hierarchical all-reduce (issue #12)
# on CSCS Alps "beverin". Real RCCL over the Slingshot/CXI fabric via the
# tokenspeed-rocm-aiter-myofi container (RCCL + from-source aws-ofi-nccl).
#
# Submit from the repo root on beverin:
#   REPO=$PWD sbatch slurm/bench_allreduce_beverin.sbatch
#
#SBATCH --job-name=xk-allreduce
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=allreduce-%j.out
#SBATCH --error=allreduce-%j.out

set -euo pipefail

REPO="${REPO:-$PWD}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29577}"
# Confirm RCCL sees the 4+4 xGMI islands rather than silently doing a flat 8-way.
export NCCL_TOPO_DUMP_FILE="${REPO}/rccl-topo-%j.xml"

echo "REPO=$REPO ENV=$ENV_NAME MASTER_ADDR=$MASTER_ADDR nodes=$SLURM_NNODES tasks=$SLURM_NTASKS"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  # All 4 GPUs visible per task; the script picks its device by SLURM_LOCALID.
  unset ROCR_VISIBLE_DEVICES || true
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  python -u "'"$REPO"'/benchmarks/bench_hierarchical_all_reduce.py" \
      --ranks-per-node 4 --hidden 7168 --sizes 1 2 4 8 16 --iters 100 --warmup 30
'
