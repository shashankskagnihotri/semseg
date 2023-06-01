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

def scale_process(image, classes=21, crop_h=256, crop_w=256, h=0, w=0, mean=None, std=None, stride_rate=2/3, paths=None):
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
            #image_crop = image[s_h:e_h, s_w:e_w].copy()
            
            image[s_h:e_h, s_w:e_w] = cv2.imread(paths[count])
            count+=1
            if count ==6:                
                return image, count
            
    return image_crop, count


path_mask_gt = "/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/SegmentationClassAug/2010_001995.png"
path_mask_pred ="/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_fgsm/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/gray/2010_001995.png"
path_mask_seg20 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/gray/2010_001995.png"
path_mask_cos20 ="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/gray/2010_001995.png"
path_colored_seg20 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/color/2010_001995.png"
path_colored_cos20 ="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/color/2010_001995.png"
path_full_image="/work/ws-tmp/sa058646-segment/semseg/dataset/voc2012/JPEGImages/2010_001995.jpg"

cos_20_path_crop1="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_1.png"
segpgd_20_path_crop1 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_1.png"
cos_20_path_crop2="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_2.png"
segpgd_20_path_crop2 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_2.png"
cos_20_path_crop3="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_3.png"
segpgd_20_path_crop3 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_3.png"
cos_20_path_crop4="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_4.png"
segpgd_20_path_crop4 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_4.png"
cos_20_path_crop5="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_5.png"
segpgd_20_path_crop5 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_5.png"
cos_20_path_crop6="/work/ws-tmp/sa058646-segment/semseg/runs/neurips/cospgd/alpha_004/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_6.png"
segpgd_20_path_crop6 = "/work/ws-tmp/sa058646-segment/semseg/runs/neurips/high_alpha_segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_40/segpgd/2010_001995_crop_6.png"




mask_gt = cv2.imread(path_mask_gt, cv2.IMREAD_GRAYSCALE)
mask_pred = cv2.imread(path_mask_pred, cv2.IMREAD_GRAYSCALE)
mask_seg20 = cv2.imread(path_mask_seg20, cv2.IMREAD_GRAYSCALE)
mask_cos20 = cv2.imread(path_mask_cos20, cv2.IMREAD_GRAYSCALE)
colored_seg20 = cv2.imread(path_colored_seg20)
colored_cos20 = cv2.imread(path_colored_cos20)
full_image = cv2.imread(path_full_image)
cos_20_paths = [cos_20_path_crop1, cos_20_path_crop2, cos_20_path_crop3, cos_20_path_crop4, cos_20_path_crop5, cos_20_path_crop6]
segpgd_20_paths = [segpgd_20_path_crop1, segpgd_20_path_crop2, segpgd_20_path_crop3, segpgd_20_path_crop4, segpgd_20_path_crop5, segpgd_20_path_crop6]
#import ipdb;ipdb.set_trace()
cos_crop_image, _ = scale_process(full_image, paths = cos_20_paths)
full_image = cv2.imread(path_full_image)
segpgd_crop_image, _ = scale_process(full_image, paths = segpgd_20_paths)
#border_image, _ = make_border(full_image)
cv2.imwrite("nips_figures/2010_001995_cos_adv_image.png", cos_crop_image)
cv2.imwrite("nips_figures/2010_001995_segpgd_adv_image.png", segpgd_crop_image)
#cv2.imwrite("nips_figures/full_border.png", border_image)
full_image = cv2.imread(path_full_image)

psnr_cos_20 = cv2.PSNR(full_image, cos_crop_image)
psnr_seg_20 = cv2.PSNR(full_image, segpgd_crop_image)

gt_col = colorize(mask_gt, colors)
gt_col.save("nips_figures/2010_001995_gt_full.png")

intersection, union, _ = intersectionAndUnion(mask_gt, mask_seg20 , 21)
seg20_full_pred_iou = np.average(intersection)/np.average(union)

intersection, union, _ = intersectionAndUnion(mask_gt, mask_cos20 , 21)
cos20_full_pred_iou = np.average(intersection)/np.average(union)

print("CosPGD PSNR: {}\nSegPGD PSNR: {}\n CosPGD mIoU: {}\n SegPGD mIoU: {}".format(psnr_cos_20, psnr_seg_20, cos20_full_pred_iou, seg20_full_pred_iou))

"""
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
"""
