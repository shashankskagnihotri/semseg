import cv2
import math
import numpy as np
from util.util import AverageMeter, intersectionAndUnion, intersectionAndUnionGPU, check_makedirs, colorize



def PSNR(original, compressed):
    mse1 = np.mean((original[0] - compressed[0]) ** 2)
    mse2 = np.mean((original[1] - compressed[1]) ** 2)
    mse3 = np.mean((original[2] - compressed[2]) ** 2)
    #if(mse1 == 0):  # MSE is zero means no noise is present in the signal .
    #              # Therefore PSNR have no importance.
    #    return 100
    mse = mse1 + mse2 +mse3
    max_pixel = 255.0
    psnr = 10 * math.log10((max_pixel**2) / mse)
    return psnr

#def main():
clean_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/input/2007_000042_crop_1.png"
clean_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/input/2007_000187_crop_1.png"

fgsm_02_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.2/fgsm/2007_000042_crop_1.png"
fgsm_02_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.2/fgsm/2007_000187_crop_1.png"

fgsm_20_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_2.0/fgsm/2007_000042_crop_1.png"
fgsm_20_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_2.0/fgsm/2007_000187_crop_1.png"

cospgd_02_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_0.2/segpgd/2007_000042_crop_1.png"
cospgd_02_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_0.2/segpgd/2007_000187_crop_1.png"

cospgd_20_input1 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_2.0/segpgd/2007_000042_crop_1.png"
cospgd_20_input2 = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/high_alpha/one_step_cospgd/alpha_015/voc2012/unet_11/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/epsilon_2.0/segpgd/2007_000187_crop_1.png"

clean_path = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/fgsm/voc2012/unet_11/correct_high_fgsm_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/eps_0.0/input/2007_009458_crop_5.png"
cos_20_path="/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/cospgd/alpha_015/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/segpgd/2007_009458_crop_5.png"
segpgd_20_path = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/segpgd/2007_009458_crop_5.png"


clean1 = cv2.imread(clean_input1)#.transpose(2, 0, 1)
clean2 = cv2.imread(clean_input2)#.transpose(2, 0, 1)
fgsm_02_1 = cv2.imread(fgsm_02_input1)#.transpose(2, 0, 1)
fgsm_02_2 = cv2.imread(fgsm_02_input2)#.transpose(2, 0, 1)
fgsm_20_1 = cv2.imread(fgsm_20_input1)#.transpose(2, 0, 1)
fgsm_20_2 = cv2.imread(fgsm_20_input2)#.transpose(2, 0, 1)
cospgd_02_1 = cv2.imread(cospgd_02_input1)#.transpose(2, 0, 1)
cospgd_02_2 = cv2.imread(cospgd_02_input2)#.transpose(2, 0, 1)
cospgd_20_1 = cv2.imread(cospgd_20_input1)#.transpose(2, 0, 1)
cospgd_20_2 = cv2.imread(cospgd_20_input2)#.transpose(2, 0, 1)

cos_20 = cv2.imread(cos_20_path)#.transpose(2, 0, 1)
seg_20 = cv2.imread(segpgd_20_path)#.transpose(2, 0, 1)
clean = cv2.imread(clean_path)

print('SHAPE: {}'.format(cospgd_02_1.shape))
print('MAX: {}'.format(np.max(cospgd_02_1)))

#colors = np.loadtxt(args.colors_path).astype('uint8'


def scale_process(image, classes=21, crop_h=240, crop_w=249, h=0, w=0, mean=None, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
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
            
    return image_crop

# psnr_fgsm_02_1 = PSNR(clean1, fgsm_02_1)
# psnr_fgsm_02_2 = PSNR(clean2, fgsm_02_2)
# psnr_fgsm_20_1 = PSNR(clean1, fgsm_20_1)
# psnr_fgsm_20_2 = PSNR(clean2, fgsm_20_2)

# psnr_cospgd_02_1 = PSNR(clean1, cospgd_02_1)
# psnr_cospgd_02_2 = PSNR(clean2, cospgd_02_2)
# psnr_cospgd_20_1 = PSNR(clean1, cospgd_20_1)
# psnr_cospgd_20_2 = PSNR(clean2, cospgd_20_2)

psnr_fgsm_02_1 = cv2.PSNR(clean1, fgsm_02_1)
psnr_fgsm_02_2 = cv2.PSNR(clean2, fgsm_02_2)
psnr_fgsm_20_1 = cv2.PSNR(clean1, fgsm_20_1)
psnr_fgsm_20_2 = cv2.PSNR(clean2, fgsm_20_2)

psnr_cospgd_02_1 = cv2.PSNR(clean1, cospgd_02_1)
psnr_cospgd_02_2 = cv2.PSNR(clean2, cospgd_02_2)
psnr_cospgd_20_1 = cv2.PSNR(clean1, cospgd_20_1)
psnr_cospgd_20_2 = cv2.PSNR(clean2, cospgd_20_2)

psnr_cos_20 = cv2.PSNR(clean, cos_20)
psnr_seg_20 = cv2.PSNR(clean, seg_20)
psnr_clean = cv2.PSNR(clean, clean)

sintel1_path ="/work/ws-tmp/sa058646-cospgd/RAFT/experiements/icml/sintel/cospgd/iterations_40/epsilon_0.03/alpha_0.15/images/input1/orig_0_clean_left.png"
sintel2_path ="/work/ws-tmp/sa058646-cospgd/RAFT/experiements/icml/sintel/cospgd/iterations_40/epsilon_0.03/alpha_0.15/images/input1/perturbed_0_clean_left.png"
sintel3_path ="/work/ws-tmp/sa058646-cospgd/RAFT/experiements/icml/sintel/cospgd/iterations_3/epsilon_0.03/alpha_0.15/images/input1/perturbed_0_clean_left.png"

sintel1 = cv2.imread(sintel1_path)
sintel2 = cv2.imread(sintel2_path)
sintel3 = cv2.imread(sintel3_path)

psnr_sintel1 = cv2.PSNR(sintel1, sintel1)
psnr_sintel1_2 = cv2.PSNR(sintel1, sintel2)
psnr_sintel1_3 = cv2.PSNR(sintel1, sintel3)

print('SINTEL itself: {}\n Sintel perturbed 40: {}\nSintel perturbed 10: {}\n'.format(psnr_sintel1, psnr_sintel1_2, psnr_sintel1_3))

print('FGSM 0.2 input 1: {}\n FGSM 0.2 input 2: {}\n FGSM 2.0 input 1: {}\n FGSM 2.0 input 2: {}\n'.format(psnr_fgsm_02_1, psnr_fgsm_02_2, psnr_fgsm_20_1, psnr_fgsm_20_2))
print('CosPGD 0.2 input 1: {}\n CosPGD 0.2 input 2: {}\n CosPGD 2.0 input 1: {}\n CosPGD 2.0 input 2: {}\n'.format(psnr_cospgd_02_1, psnr_cospgd_02_2, psnr_cospgd_20_1, psnr_cospgd_20_2))

print('CosPGD 20 iterations: {}\n SegPGD 20 iterations: {}\n Clean:{}'.format(psnr_cos_20, psnr_seg_20, psnr_clean))