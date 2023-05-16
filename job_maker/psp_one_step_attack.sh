#!/bin/bash
#SBATCH --output=slurm/neurips/final_testing/new_one_step_adv_trained_%A.out
#SBATCH --error=slurm/neurips/final_testing/new_one_step_adv_trained_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de
#SBATCH -d afterok:3110304:3110303:3110302:3110301

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/fgsm.py --config $6 config.save_folder $1 config.epsilon $2 config.iterations $3 config.alpha $4 config.attack $5

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime