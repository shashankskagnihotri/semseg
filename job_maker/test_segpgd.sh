#!/bin/bash

#backbone="convnext_tiny SLaK_tiny resnet50"
backbone="convnext_tiny"
epochs="250"
base_lr="0.0001"
#save_path="runs/new_exp_slak/voc2012/unet"
#optimizers="adamw"
#criterion="cross_entropy"
#trans_kernel="2 7 31"
#use_convnext="False True"
#backbone_kernel="3"
#small_trans="3 5"
#small_conv="5"
save_path="runs/new_exp_slak/voc2012/unet_11"
actual_save_path="runs/neurips/high_alpha_segpgd/voc2012/unet_11"
#save_path="runs/freq_upsampling_correct/voc2012/unet"
optimizers="adamw"
criterion="cross_entropy"
#trans_kernel="2 7 11"
trans_kernel="2"
#use_convnext="False True"
use_convnext="False"
#backbone_kernel="7 11"
backbone_kernel="3"
small_trans="0"
small_conv="0"
#epsilon="3 5 10 20 40 100"
epsilon="10 20 40 100"


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
		        if [[ $trans = "11" ]]
			then
			    small_trans="3"
				elif [[ $trans = "31" ]]
			then
				small_trans="5"
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
					if [[ $trans = "31" ]]
					then
						backbone_kernel="7 31"
					else
				    	backbone_kernel="7 11"
					fi
				fi
				for bk in $backbone_kernel
				do
				    if [[ $bk = "11" ]]
				    then
				        small_conv="3"
					elif [[ $bk = "31" ]]
					then
						small_conv="5"
				    else
					small_conv="0"
				    fi
		                    for eps in $epsilon
							do
								path="${save_path}/${net}_trans_kernel_${trans}_small_trans_${st}_convnext_backbone_${use_convx}_backbone_kernel_${bk}_small_conv_${small_conv}/model"
		                    	folder="${actual_save_path}/correct_segpgd_attack/${net}_trans_kernel_${trans}_small_trans_${st}_convnext_backbone_${use_convx}_backbone_kernel_${bk}_small_conv_${small_conv}/val/ss/iterations_${eps}"
								var=$(ls ${path}| sort -V | tail -n 1)
								model_location="${path}/${var}"
		                    	env
				    			sbatch -p gpu -t 23:59:59 --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=150G job_maker/unet_voc_segpgd.sh $net $epoch $lr $path $folder $optimizer $criteria $trans $bk $st $small_conv $use_convx $model_location $eps
							done
				done
			    done
			done
		    done
                done
            done
        done
    done
done