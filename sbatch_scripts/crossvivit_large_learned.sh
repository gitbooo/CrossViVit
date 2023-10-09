#!/bin/bash
#SBATCH --job-name=crossvivit_large_learned
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH -o use_your_path

module --quiet load anaconda/3
conda activate ocf

python main.py \
experiment=cross_vivit \
datamodule.num_workers=4 \
datamodule.batch_size=10 \
seed=42 \
resume=True \
pl_module.model.ts_masking_ratio=0 \
pl_module.model.ctx_masking_ratio=0.99 \
pl_module.criterion._target_=torch.nn.L1Loss \
pl_module.model.depth=16 \
pl_module.model.ctx_channels=18 \
pl_module.model.dim=384 \
pl_module.use_dp=False \
pl_module.model.heads=12 \
pl_module.model.mlp_ratio=4 \
pl_module.model.pe_type=learned \
trainer.strategy=ddp \
trainer.max_epochs=100
