#!/bin/bash
#SBATCH --output=slurm/neurips/segpgd_attack/new_segpgd_1_%A.out
#SBATCH --error=slurm/neurips/segpgd_attack/new_segpgd_1_%A.err
#SBATCH --mail-type=ALL
#SBATCH --exclude=gpu-node006
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/segpgd.py config.backbone $1 config.epochs $2 config.base_lr $3 config.save_path $4 config.save_folder $5 config.optimizer $6 config.criterion $7 config.trans_kernel $8 config.backbone_kernel $9 config.small_trans ${10} config.small_conv ${11} config.use_convnext_backbone ${12} config.model_path ${13} config.iterations ${14} config.alpha 0.01

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
