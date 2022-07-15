#!/bin/bash

backbone="resnet50 convnext_tiny none"
epochs="50 100 250 500"
base_lr="0.001 0.0001 0.00001"
save_path="new_exp/voc2012/unet"
optimizers="adam sgd"
criterion="mse cross_entropy"

for epoch in $epochs
do
    for lr in $base_lr
    do
        for optimizer in $optimizers
        do
            for criteria in $criterion
            do
                for net in $backbone
                do
                    path="${save_path}_${epoch}_${net}_lr_${lr}_${optmizer}_${criteria}/model"
                    folder="${save_path}_${epoch}_${net}_lr_${lr}_${optmizer}_${criteria}/val"
                    sbatch -p gpu -t 23:59:59 --gres=gpu:2 --cpus-per-task=64 --mem=150G unet_voc.sh $net $epoch $lr $path $folder $optimizer $criteria
                done
            done
        done
    done
done