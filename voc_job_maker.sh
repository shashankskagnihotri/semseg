#!/bin/bash

backbone="convnext_tiny SLaK_tiny resnet50"
epochs="250"
base_lr="0.0001"
save_path="new_exp_slak/voc2012/unet"
optimizers="adamw"
criterion="cross_entropy"
trans_kernel="2 7 31"
use_convnext="False True"
backbone_kernel="3"
small_trans="3 5"
small_conv="5"

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
		    for trans in $trans_kernel
		    do
		        if [[ $trans = "31" ]]
			then
			    small_trans="3 5"
			else
			    small_trans="0"
			fi
			for st in $small_trans
			do
			    for use_convx in $use_convnext
			    do
			        if [[ $use_convx = "False" ]]
				then
				    backbone_kernel="3"
				    small_conv="0"
				else
				    backbone_kernel="3 7 31"
				fi
				for bk in $backbone_kernel
				do
				    if [[ $bk = "31" ]]
				    then
				        small_conv="5"
				    else
					small_conv="0"
				    fi
		                    path="${net}_trans_kernel_${trans}_small_trans_${st}_convnext_backbone_${use_convx}_backbone_kernel_${bk}_small_conv_${small_conv}/model"
		                    folder="${net}_trans_kernel_${trans}_small_trans_${st}_convnext_backbone_${use_convx}_backbone_kernel_${bk}_small_conv_${small_conv}/val"
		                    env
				    sbatch -p gpu -t 23:59:59 --gres=gpu:1 --ntasks=1 --cpus-per-task=32 --mem=150G unet_voc.sh $net $epoch $lr $path $folder $optimizer $criteria $trans $bk $st $small_conv $use_convx
				done
			    done
			done
		    done
                done
            done
        done
    done
done
