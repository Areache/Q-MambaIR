#!/bin/bash

# Example SLURM script — adjust partition/account/paths to match your cluster.
# For non-SLURM environments, run the python command directly.

#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --gpus=2
#SBATCH --mem=64GB
#SBATCH --job-name=Ours_4b_2_x4_20k
#SBATCH --output=/path/to/logs/Ours_4b_2_x4_20k.out
#SBATCH --error=/path/to/logs/Ours_4b_2_x4_20k.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --account=YOUR_ACCOUNT

# Set REPO_ROOT to the directory where you cloned Q-MambaIR
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
nvidia-smi

python -m torch.distributed.launch --nproc_per_node=2 --master_port=2488 \
  $REPO_ROOT/basicsr/train.py \
  -opt $REPO_ROOT/options/train/Ours/x4/quant_train_bl_lightSR_x4_4b.yml \
  --launcher pytorch