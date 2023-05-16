#!/bin/bash

#iterations="3 5 10 20 40 100"
iterations="5 10 20 40"
epsilon="0.03"
#attacks="segpgd cospgd"
attacks="cospgd"
save_folder="exp/voc2012/pspnet50/result/epoch_50/val/breakdown10"
alpha="0.15"

for it in $iterations
do
    for attack in $attacks
    do  
        path="${save_folder}/multi_step/${attack}/${eps}/${it}"
        sbatch -p gpu -t 23:59:59 --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=55G job_maker/10attack.sh $path $epsilon $it $alpha $attack
    done
done
