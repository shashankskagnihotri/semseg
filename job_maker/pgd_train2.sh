#!/bin/bash

#iterations="3 5 10 20 40 100"
#iterations="5 10 20 40 100"
iterations="3 5"
epsilon="0.03"
#attacks="segpgd cospgd"
attacks="cospgd"
save_folder="exp/voc2012/pspnet50/icml_rebutal/training_adv_third_cospgd_alpha_0.01"
alpha="0.01"

for it in $iterations
do
    for attack in $attacks
    do  
        path="${save_folder}/multi_step/${attack}/${eps}/${it}"
        job_name="${alpha}_${it}_${attack}"
        sbatch -p gpu -t 23:59:59 --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=70G -J ${job_name} job_maker/psp_multi_step_training.sh $path $epsilon $it $alpha $attack
        echo $job_name
    done
done
