#!/bin/bash
#SBATCH --output=slurm/neurips/adv_training/new_seg_cospgd_%A.out
#SBATCH --error=slurm/neurips/adv_training/new_seg_cospgd_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/train_adv.py config.save_path $1 config.save_folder $1 config.epsilon ${2} config.iterations ${3} config.alpha ${4} config.attack ${5} config.epochs 50

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime