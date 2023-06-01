import cv2
import math
import numpy as np
import sys
import os
file_dir = os.path.dirname("/work/ws-tmp/sa058646-segment/semseg/util")
sys.path.append(file_dir)
from util.util import AverageMeter, intersectionAndUnion, intersectionAndUnionGPU, check_makedirs, colorize

colors = np.loadtxt('data/voc2012/voc2012_colors.txt').astype('uint8')
#intersection, union, target = intersectionAndUnion(pred, target, 21)

def make_border(image, classes=21, crop_h=256, crop_w=256, h=0, w=0, mean=None, std=None, stride_rate=2/3):
    if len(image.shape) == 3:
        ori_h, ori_w, _ = image.shape
    else:
        ori_h, ori_w = image.shape
    count = 0
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    if len(image.shape) == 3:
        new_h, new_w, _ = image.shape
    else:
        new_h, new_w = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            border = cv2.copyMakeBorder(image_crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 255])
            #image_crop = image[s_h:e_h, s_w:e_w].copy()
            tmp = image.copy()            
            count+=1
            if count ==1:
                #import ipdb;ipdb.set_trace()
                for it in range(3):
                    tmp[s_h:e_h+2, s_w-2:e_w+2,it] = border[::,it]
                    #return s_h, e_h, s_w, e_w
            
    #return s_h, e_h, s_w, e_w
    return tmp

def scale_process(image, classes=21, crop_h=256, crop_w=256, h=0, w=0, mean=None, std=None, stride_rate=2/3):
    if len(image.shape) == 3:
        ori_h, ori_w, _ = image.shape
    else:
        ori_h, ori_w = image.shape
    count = 0
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:        
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    if len(image.shape) == 3:
        new_h, new_w, _ = image.shape
    else:
        new_h, new_w = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count+=1
            if count ==1:
                return image_crop, count
            
    return image_crop, count

# path_mask_gt = "/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/SegmentationClassAug/2011_000482.png"
# path_mask_pred ="/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_fgsm/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/gray/2011_000482.png"
# path_mask_seg20 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/gray/2011_000482.png"
# path_mask_cos20 ="/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/cospgd/alpha_015/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/gray/2011_000482.png"
# path_colored_seg20 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/color/2011_000482.png"
# path_colored_cos20 ="/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/cospgd/alpha_015/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/color/2011_000482.png"
# path_full_image="/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/JPEGImages/2011_000482.jpg"

# cos_20_path="/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/cospgd/alpha_015/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/segpgd/2011_000482_crop_2.png"
# segpgd_20_path = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/segpgd/2011_000482_crop_2.png"



path_mask_gt_input1 = "/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/SegmentationClassAug/2007_000042.png"
path_mask_gt_input2 = "/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/SegmentationClassAug/2007_000187.png"

clean_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/input/2007_000042_crop_1.png"
clean_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/input/2007_000187_crop_1.png"

fgsm_02_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.2/fgsm/2007_000042_crop_1.png"
pred_fgsm_02_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.2/gray/2007_000042.png"
fgsm_02_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.2/fgsm/2007_000187_crop_1.png"
pred_fgsm_02_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.2/gray/2007_000187.png"

pred_fgsm_20_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_2.0/gray/2007_000042.png"
pred_fgsm_20_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_2.0/gray/2007_000187.png"

fgsm_20_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_2.0/fgsm/2007_000042_crop_1.png"
fgsm_20_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_2.0/fgsm/2007_000187_crop_1.png"

cospgd_02_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_0.2/segpgd/2007_000042_crop_1.png"
cospgd_02_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_0.2/segpgd/2007_000187_crop_1.png"

cospgd_20_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_2.0/segpgd/2007_000042_crop_1.png"
cospgd_20_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_2.0/segpgd/2007_000187_crop_1.png"

pred_cospgd_02_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_0.2/gray/2007_000042.png"
pred_cospgd_02_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_0.2/gray/2007_000187.png"

pred_cospgd_20_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_2.0/gray/2007_000042.png"
pred_cospgd_20_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_2.0/gray/2007_000187.png"

path_iccv_gt = "/work/ws-tmp/sa058646-segment/semseg/runs/ground_truth_images/2007_000033.png"
path_iccv_gt2 = "/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/SegmentationClassAug/2007_000033.png"

#iccv_gt = cv2.imread(path_iccv_gt, cv2.IMREAD_GRAYSCALE)

#gt_input1, _ = scale_process(cv2.imread(path_iccv_gt, cv2.IMREAD_GRAYSCALE))
#gt_input2, _ = scale_process(cv2.imread(path_iccv_gt2, cv2.IMREAD_GRAYSCALE))
#colorize(gt_input1, colors).save('iccv1.png')
"""
iccv_gt2 = cv2.imread(path_iccv_gt2, cv2.IMREAD_GRAYSCALE)
colorize(iccv_gt2, colors).save('iccv_full.png')
"""
"""
mask_gt = cv2.imread(path_mask_gt, cv2.IMREAD_GRAYSCALE)
mask_pred = cv2.imread(path_mask_pred, cv2.IMREAD_GRAYSCALE)
mask_seg20 = cv2.imread(path_mask_seg20, cv2.IMREAD_GRAYSCALE)
mask_cos20 = cv2.imread(path_mask_cos20, cv2.IMREAD_GRAYSCALE)
colored_seg20 = cv2.imread(path_colored_seg20)
colored_cos20 = cv2.imread(path_colored_cos20)
full_image = cv2.imread(path_full_image)


mask_gt_input1, _ = scale_process(cv2.imread(path_mask_gt_input1, cv2.IMREAD_GRAYSCALE))
mask_gt_input2, _ = scale_process(cv2.imread(path_mask_gt_input2, cv2.IMREAD_GRAYSCALE))

mask_fgsm_02_input1, _ = scale_process(cv2.imread(pred_fgsm_02_input1, cv2.IMREAD_GRAYSCALE))
mask_fgsm_02_input2, _ = scale_process(cv2.imread(pred_fgsm_02_input2, cv2.IMREAD_GRAYSCALE))
mask_fgsm_20_input1, _ = scale_process(cv2.imread(pred_fgsm_20_input1, cv2.IMREAD_GRAYSCALE))
mask_fgsm_20_input2, _ = scale_process(cv2.imread(pred_fgsm_20_input2, cv2.IMREAD_GRAYSCALE))

mask_cospgd_02_input1, _ = scale_process(cv2.imread(pred_cospgd_02_input1, cv2.IMREAD_GRAYSCALE))
mask_cospgd_02_input2, _ = scale_process(cv2.imread(pred_cospgd_02_input2, cv2.IMREAD_GRAYSCALE))
mask_cospgd_20_input1, _ = scale_process(cv2.imread(pred_cospgd_20_input1, cv2.IMREAD_GRAYSCALE))
mask_cospgd_20_input2, _ = scale_process(cv2.imread(pred_cospgd_20_input2, cv2.IMREAD_GRAYSCALE))

mIoU = {}
intersection, union, _ = intersectionAndUnion(mask_gt_input1, mask_fgsm_02_input1 , 21)
mIoU['fgsm_02_input1'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input2, mask_fgsm_02_input2 , 21)
mIoU['fgsm_02_input2'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input1, mask_fgsm_20_input1 , 21)
mIoU['fgsm_20_input1'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input2, mask_fgsm_20_input2 , 21)
mIoU['fgsm_20_input2'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input1, mask_cospgd_02_input1 , 21)
mIoU['cospgd_02_input1'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input2, mask_cospgd_02_input2 , 21)
mIoU['cospgd_02_input2'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input1, mask_cospgd_20_input1 , 21)
mIoU['cospgd_20_input1'] = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt_input2, mask_cospgd_20_input2 , 21)
mIoU['cospgd_20_input2'] = np.average(intersection)/np.average(union)


colorize(mask_fgsm_02_input1, colors).save('figure4_masks/fgsm_02_input1.png')
colorize(mask_fgsm_02_input2, colors).save('figure4_masks/fgsm_02_input2.png')
colorize(mask_fgsm_20_input1, colors).save('figure4_masks/fgsm_20_input1.png')
colorize(mask_fgsm_20_input2, colors).save('figure4_masks/fgsm_20_input2.png')
colorize(mask_cospgd_02_input1, colors).save('figure4_masks/cospgd_02_input1.png')
colorize(mask_cospgd_02_input2, colors).save('figure4_masks/cospgd_02_input2.png')
colorize(mask_cospgd_20_input1, colors).save('figure4_masks/cospgd_20_input1.png')
colorize(mask_cospgd_20_input2, colors).save('figure4_masks/cospgd_20_input2.png')

print(mIoU)
"""

path_mask_gt = "/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/SegmentationClassAug/2011_001292.png"
path_mask_pred ="/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_fgsm/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/gray/2011_001292.png"
path_mask_seg20 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/gray/2011_001292.png"
path_mask_cos20 ="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/gray/2011_001292.png"
path_colored_seg20 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/color/2011_001292.png"
path_colored_cos20 ="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/color/2011_001292.png"
path_full_image="/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/JPEGImages/2011_001292.jpg"

cos_20_path="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2011_001292_crop_1.png"
segpgd_20_path = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2011_001292_crop_1.png"





mask_gt = cv2.imread(path_mask_gt, cv2.IMREAD_GRAYSCALE)
mask_pred = cv2.imread(path_mask_pred, cv2.IMREAD_GRAYSCALE)
mask_seg20 = cv2.imread(path_mask_seg20, cv2.IMREAD_GRAYSCALE)
mask_cos20 = cv2.imread(path_mask_cos20, cv2.IMREAD_GRAYSCALE)
colored_seg20 = cv2.imread(path_colored_seg20)
colored_cos20 = cv2.imread(path_colored_cos20)
full_image = cv2.imread(path_full_image)


crop_image, _ = scale_process(full_image)
#border_image, _ = make_border(full_image)
cv2.imwrite("nips_figures/crop_image.png", crop_image)
cv2.imwrite("nips_figures/full_image.png", full_image)
#cv2.imwrite("nips_figures/full_border.png", border_image)

gt_crop, count = scale_process(mask_gt)
gt_crop_col = colorize(gt_crop, colors)
gt_crop_col.save("nips_figures/gt_crop.png")
gt_col = colorize(mask_gt, colors)
gt_col.save("nips_figures/gt_full.png")

pred_crop, count = scale_process(mask_pred)
pred_crop_col = colorize(gt_crop, colors)
pred_crop_col.save("nips_figures/pred_crop.png")
pred_col = colorize(mask_pred, colors)
pred_col.save("nips_figures/pred_full.png")

seg20_crop, count = scale_process(mask_seg20)
seg20_crop_col = colorize(seg20_crop, colors)
seg20_crop_col.save("nips_figures/seg40_crop.png")

cos20_crop, count = scale_process(mask_cos20)
cos20_crop_col = colorize(cos20_crop, colors)
cos20_crop_col.save("nips_figures/cos40_crop.png")

intersection, union, _ = intersectionAndUnion(mask_gt, mask_pred , 21)
full_pred_iou = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(gt_crop, pred_crop , 21)
#import ipdb;ipdb.set_trace()
pred_iou = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(gt_crop, seg20_crop , 21)
seg20_iou = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(gt_crop, cos20_crop , 21)
cos20_iou = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(gt_crop, gt_crop , 21)
sanity = np.average(intersection)/np.average(union)

cos_20 = cv2.imread(cos_20_path)#.transpose(2, 0, 1)
seg_20 = cv2.imread(segpgd_20_path)

cv2.imwrite("nips_figures/seg40_image.png", seg_20)
cv2.imwrite("nips_figures/cos40_image.png", cos_20)

#import ipdb;ipdb.set_trace()
psnr_cos_20 = cv2.PSNR(crop_image, cos_20)
psnr_seg_20 = cv2.PSNR(crop_image, seg_20)

print("original pred: {}\n pred on crop: {}\n Seg20: {}\n Cos20: {}\n Sanity:{}".format(full_pred_iou, pred_iou, seg20_iou, cos20_iou, sanity))
print("PSNR SegPGD 20: {}\n PSNR CosPGD 20:{}".format(psnr_seg_20, psnr_cos_20))

#import ipdb;ipdb.set_trace()
#cv2.imwrite('test.png', cv2.cvtColor(gt_crop, cv2.COLOR_RGB2BGR))

print(count)

