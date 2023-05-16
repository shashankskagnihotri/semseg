#!/bin/bash
#SBATCH --time=12:59:59
#SBATCH --nodes=1
#SBATCH --mem 70G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --output=slurm/neurips/adv_training/1new_continued_%A_%a.out
#SBATCH --error=slurm/neurips/adv_training/1new_continued_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de

#SBATCH -J PSP_CONT_TRAIN


echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
start=`date +%s`


python tool/train_adv.py --config $1

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime