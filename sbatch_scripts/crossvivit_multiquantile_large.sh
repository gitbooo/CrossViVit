#!/bin/bash
#SBATCH --job-name=crossvivit_multiquantile_large
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=10                            
#SBATCH --gres=gpu:rtx8000:2                                   
#SBATCH --mem=80G                                       
#SBATCH --time=48:00:00                           
#SBATCH -o use_your_own_path

module --quiet load anaconda/3
conda activate ocf

python main.py \
experiment=cross_vivit_quantile_mask \
datamodule.num_workers=4 \
datamodule.batch_size=10 \
seed=42 \
pl_module.cutoff_epoch=10000 \
+pl_module.criterion.quantiles="[0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]" \
+pl_module.criterion.quantile_weights="[0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]" \
resume=True \
pl_module.model.num_mlp_heads=11 \
pl_module.model.ts_masking_ratio=0 \
pl_module.model.ctx_masking_ratio=0.99 \
pl_module.model.depth=16 \
pl_module.model.dim=384 \
pl_module.model.heads=12 \
pl_module.model.mlp_ratio=4 \
pl_module.optimizer.lr=0.0016 \
pl_module.use_dp=False \
pl_module.model.ctx_channels=18 \
trainer.max_epochs=100 \
trainer.strategy=ddp 


