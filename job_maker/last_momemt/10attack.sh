#!/bin/bash
#SBATCH --output=slurm/icml/break_down/10th_120_%A.out
#SBATCH --error=slurm/icml/break_down/10th_120_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/fgsm.py config.save_folder $1 config.epsilon ${2} config.iterations ${3} config.alpha ${4} config.attack ${5} config.index_start 1080 config.index_end 1200

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime