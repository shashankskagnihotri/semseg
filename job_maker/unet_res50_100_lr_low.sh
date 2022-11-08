#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python tool/train.py --config config/voc2012/voc2012_unet50_100.yaml config.base_lr 0.00001 config.save_path exp/voc2012/unet101_resnet_pretrained_lr_0.00001/model

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime