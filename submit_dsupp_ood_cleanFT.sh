#!/bin/bash
#SBATCH --job-name=dsupp_ood_cleanFT
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/storage/work/axm6766/gfm-rag-outputs/dsupp_ood_cleanFT_%j.log
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|a40|v100'
#SBATCH --exclude=p-gc-3002,p-gc-3003
module load cuda/12.6.0
module load gcc/12.2.0
cd /storage/home/axm6766/projects/gfm-rag2.0/gfm-rag
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HYDRA_FULL_ERROR=1 /storage/work/axm6766/.conda/envs/gfmrag/bin/python -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner --config-name sft_training_pilotData2_victim_distinctsupp_cleanFT
