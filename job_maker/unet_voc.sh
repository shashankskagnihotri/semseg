#!/bin/bash
#SBATCH --output=slurm/rebuttal/training_%A.out
#SBATCH --error=slurm/rebuttal/training_%A.err

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/train.py config.backbone $1 config.epochs $2 config.base_lr $3 config.save_path $4 config.save_folder $5 config.optimizer $6 config.criterion $7 config.trans_kernel $8 config.backbone_kernel $9 config.small_trans ${10} config.small_conv ${11} config.use_convnext_backbone ${12} config.start_epoch 0

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime