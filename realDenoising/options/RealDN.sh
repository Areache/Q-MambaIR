#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2
#SBATCH --mem=64GB
#SBATCH --job-name=RealDN_ours_2b
#SBATCH --output=/leonardo/home/userexternal/ychen004/QuantIR/realDenoising/logs/DN/RealDN_ours_2b.out
#SBATCH --error=/leonardo/home/userexternal/ychen004/QuantIR/realDenoising/logs/DN/RealDN_ours_2b.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrb_fm-eeg24

export PYTHONPATH=$PYTHONPATH:/leonardo/home/userexternal/ychen004/QuantIR/realDenoising
nvidia-smi

/leonardo/home/userexternal/ychen004/anaconda3/envs/mambair/bin/python -m \
torch.distributed.launch --nproc_per_node=2 --master_port=2488 \
/leonardo/home/userexternal/ychen004/QuantIR/basicsr/train.py -opt \
/leonardo/home/userexternal/ychen004/QuantIR/realDenoising/options/train_MambaIR_RealDN.yml \
--launcher pytorch

# /leonardo/home/userexternal/ychen004/anaconda3/envs/mambaIR/bin/python \
# basicsr/train.py -opt options/train/quant_train_MambaIR_lightSR_x2.yml