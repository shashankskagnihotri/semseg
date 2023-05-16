#!/bin/bash
#SBATCH --output=slurm/neurips/final_testing/0testing_low_alpha_%A.out
#SBATCH --error=slurm/neurips/final_testing/0testing_low_alpha_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de
#SBATCH --nodelist=gpu-node010


echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/fgsm.py --config $6 config.save_folder $1 config.epsilon ${2} config.iterations ${3} config.alpha ${4} config.attack ${5}

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime

## #SBATCH -d afterok:3110304:3110303:3110302:3110301