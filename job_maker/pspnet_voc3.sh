#!/bin/bash
#SBATCH --output=slurm/pspnet/kernel/training_150_%A.out
#SBATCH --error=slurm/pspnet/kernel/training_150_%A.err

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/test_convnext.py --config config/voc2012/test_voc2012_psp50.yaml config.psp_kernel 0 config.save_path runs/pspnet/voc2012/resnet_kernels/kernel_0/model config.save_folder runs/pspnet/voc2012/resnet_kernels_testing/kernel_0/val/ss config.model_path runs/pspnet/voc2012/resnet_kernels/kernel_0/model/best_model.pth

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
