#!/bin/bash
#SBATCH --job-name=crossvivit_tuning
#SBATCH --partition=long
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=1-10
#SBATCH -o use_your_path

module --quiet load anaconda/3
conda activate ocf

CUDA_VISIBLE_DEVICES=0 python main.py -m hparams_search=cross_vivit experiment=cross_vivit seed=42 resume=True