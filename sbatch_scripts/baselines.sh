#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=1-7
#SBATCH -o use_your_path

param_store=sbatch_scripts/baselines.txt
baseline=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

#SBATCH --job-name="${baseline}"

module --quiet load anaconda/3
conda activate ocf

python main.py \
experiment=$baseline \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=42 \
resume=True \
trainer.strategy=ddp \
trainer.max_epochs=250