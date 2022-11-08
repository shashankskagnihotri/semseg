#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/accuracy/training_%A.out
#SBATCH --error=slurm/accuracy/training_%A.err

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/train.py

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime

