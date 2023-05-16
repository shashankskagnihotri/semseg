#!/bin/bash

iterations="1"
epsilon="0.0 0.004 0.008 0.012 0.0157 0.0196 0.0235 0.0275 0.03 0.1 0.2 0.3 1.0"
attacks="fgsm cospgd"
#attacks="cospgd"
save_folder="exp/voc2012/pspnet50/new_neurips/final_attack_adv_trained_and_not_trained/"
alpha="0.07"
config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd.yaml"
#models="cosPGD3AT segPGD3AT cosPGD5AT segPGD5AT"
models="cosPGD3AT_0.01 cosPGD5AT_0.01 segPGD3AT segPGD5AT baseline"

for eps in $epsilon
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
            path="${save_folder}/one_step/${attack}/${model}/${eps}"
            job_name="${model}_${attack}_${eps}"
            sbatch -p gpu -t 7:59:59 --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=70G -J ${job_name} job_maker/psp_one_step_attack.sh $path $eps $iterations $alpha $attack $config
            echo "path: ${path}   epsilon: ${eps}    iterations: ${iterations}  alpha: ${eps}   attack: ${attack} config: ${config}"
        done
    done
done
