#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
start=`date +%s`

#configs=("/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd3at_0.01_voc2012_pspnet50.yaml" "/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd3at_0.05_voc2012_pspnet50.yaml" "/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd5at_0.01_voc2012_pspnet50.yaml" "/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd5at_0.05_voc2012_pspnet50.yaml")

configs="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd5at_0.01_voc2012_pspnet50.yaml /work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_segpgd5at_voc2012_pspnet50.yaml /work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd3at_0.01_voc2012_pspnet50.yaml /work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_segpgd3at_voc2012_pspnet50.yaml"
##configs=("/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd3at_voc2012_pspnet50.yaml" "/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd5at_voc2012_pspnet50.yaml")

#config=${configs[$SLURM_ARRAY_TASK_ID]}

#config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/continue_cospgd3at_0.01_voc2012_pspnet50.yaml"

for config in $configs
do
    sbatch job_maker/psp_multi_step_training_cont.sh $config
done
## python tool/train_adv.py --config $config

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime