#!/bin/bash

iterations="1"
epsilon="0.0"
attacks="fgsm"
#attacks="cospgd"
save_folder="exp/voc2012/pspnet50/icml_rebutal/adv_trained/"
alpha="0.15"
config="/work/ws-tmp/sa058646-segment/PSPNet/config/voc2012/voc2012_pspnet50_cospgd.yaml"
models="cosPGD3AT segPGD3AT"

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
            path="${save_folder}/one_step/${attack}/${model}/${eps}"
            sbatch -p gpu -t 23:59:59 --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=70G job_maker/psp_one_step_attack.sh $path $eps $iterations $eps $attack $config
            echo "path: ${path}   epsilon: ${eps}    iterations: ${iterations}  alpha: ${eps}   attack: ${attack} condif: ${config}"
        done
    done
done
