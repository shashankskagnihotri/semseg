#!/bin/bash

#iterations="3 5 10 20 40 100"
iterations="10 100"
#iterations="3"
#iterations="5 10 20 40 100"
#iterations="3 5 7"
epsilon="0.03"
#attacks="segpgd cospgd"
#attacks="segpgd pgd"
attacks="pgd"
save_folder="exp/voc2012/pspnet50/new_neurips/testing_low_alpha_pgd"
alpha="0.01"
config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd.yaml"
#models="cosPGD3AT segPGD3AT cosPGD5AT segPGD5AT cosPGD3AT_0.01 cosPGD3AT_0.05 cosPGD5AT_0.01 cosPGD5AT_0.05"
#models="cosPGD3AT_0.01 cosPGD5AT_0.01 segPGD3AT segPGD5AT baseline"
models="baseline"

for it in $iterations
do
    for attack in $attacks
    do  
        for model in $models
        do
            if [[ $model = "cosPGD3AT" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd.yaml"
            fi
            if [[ $model = "segPGD3AT" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_segpgd.yaml"
            fi
            if [[ $model = "cosPGD5AT" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd5AT.yaml"
            fi
            if [[ $model = "segPGD5AT" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_segpgd5AT.yaml"
            fi
            if [[ $model = "cosPGD3AT_0.01" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd3AT_0.01.yaml"
            fi
            if [[ $model = "cosPGD5AT_0.01" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd5AT_0.01.yaml"
            fi
            if [[ $model = "cosPGD3AT_0.05" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd3AT_0.05.yaml"
            fi
            if [[ $model = "cosPGD5AT_0.05" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd5AT_0.05.yaml"
            fi
            if [[ $model = "baseline" ]]
            then
                config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50.yaml"
            fi
            path="${save_folder}/multi_step/${attack}/${model}/${eps}/${it}"
            job_name="${model}_${attack}_${it}"
            sbatch -p gpu -t 23:59:59 --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=70G -J ${job_name} job_maker/psp_multi_step_attack.sh $path $epsilon $it $alpha $attack $config
        done
    done
done
