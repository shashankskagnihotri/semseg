import glob
import os
from pathlib import Path
import pandas as pd

path_dir = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_extra/final_fgsm_attack"
csv_file = "/work/ws-tmp/sa058646-segment/semseg/cvpr_rebuttal_convnext.csv"


path_logs = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_extra/final_fgsm_attack/**/val/ss/**/log.log"

all_logs = glob.glob(path_logs)

csv_dict={}
trans_cov_kernel = []
trans_small_kernel = []
use_convnext = []
backbone_kernel = []
small_backbone = []
eps = []
mIoUs = []
mAccs = []
allAccs = []


for path in all_logs:
    with open(path) as log:
        for line in log:
            if 'Eval result: mIoU/mAcc/allAcc' in line:
                results = line.split("Eval result: mIoU/mAcc/allAcc ",1)[1]
                mIoU = results.split('/',1)[0]
                mAcc = results.split('/',1)[1].split('/',1)[0]
                allAcc = results.split('/',1)[1].split('/',1)[1].replace('.\n','')
                #print("Results: {}\n mIoU: {}\n mAcc: {}\n allAcc:{}".format(results, mIoU, mAcc, allAcc))
                path = Path(path)
                ep = path.parent.absolute().name.replace('eps_', '')
                network = path.parent.parent.parent.parent.absolute().name
                trans_cov = network.split('convnext_tiny_trans_kernel_',1)[1][:2].replace('_','')
                trans_small = network.split('small_trans',1)[1][:2].replace('_','')
                convnext = network.split('convnext_backbone_',1)[1][:5].replace('_','')
                backbone = network.split('backbone_kernel_',1)[1][:2].replace('_','')
                small_conv = network.split('small_conv_',1)[1].replace('_','')
                #import ipdb;ipdb.set_trace()
                trans_cov_kernel.append(trans_cov)#
                trans_small_kernel.append(trans_small)
                use_convnext.append(convnext)
                backbone_kernel.append(backbone)
                small_backbone.append(small_conv)
                eps.append(ep)
                mIoUs.append(mIoU)
                mAccs.append(mAcc)
                allAccs.append(allAcc)

csv_dict = {"transposed conv kernel": trans_cov_kernel, "small parallel trans kernel": trans_small, "use ConNeXt backbone": use_convnext,
            "backbone convolution kernel": backbone_kernel, "parallel small conv backbone": small_backbone,
            "epsilon": eps, "mIoU":mIoUs, "mAcc": mAccs, "allAcc": allAccs}
import ipdb;ipdb.set_trace()
csv_df = pd.DataFrame.from_dict(csv_dict)
csv_df.to_csv(csv_file)

print(csv_df)


