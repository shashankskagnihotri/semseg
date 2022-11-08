import cv2
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
from util.util import check_makedirs

colors = ['dimgrey', 'lightcoral', 'aqua', 'gold', 'darkgreen', 'red', 'palegreen', 'dodgerblue', 'gold', 'navy', 'darkmagenta', 'lightgreen', 'darkred', 'olive', 'indigo', 'tan']
count = 0

#parent_dir = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet/**/val/ss/frequency/*.pt"
#parent_dir = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_plots_full_2_correct/**/val/ss/frequency/*.pt"
#save_path = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_plots_full_2_correct/convnext_tiny_freq_comparison_avg_channels_500.png"
#/work/ws-tmp/sa058646-segment/semseg/runs/pspnet/voc2012/resnet/testing_725/val/ss
parent_dir = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_binary_plots_full_2_correct/**/val/ss/frequency/*.pt"
#parent_dir = "/work/ws-tmp/sa058646-segment/semseg/runs/pspnet/voc2012/resnet_kernels_testing/**/val/ss/frequency/*.pt"
#parent_dir = "/work/ws-tmp/sa058646-segment/semseg/runs/freq_upsampling_correct/voc2012/unet/**/val/ss/frequency/*.pt"
#save_path = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_binary_plots_full_2_correct/convnext_tiny_freq_comparison_avg_channels_crp.png"
#save_path = "/work/ws-tmp/sa058646-segment/semseg/runs/pspnet/voc2012/resnet_kernels_testing/pspnet_resnet_freq_comparison.png"
save_path = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_binary_plots_full_2_correct/paper_plots/unet_freq_up_freq_comparison_norm_best.png"
makedir="/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_11_binary_plots_full_2_correct/paper_plots/"
check_makedirs(makedir)
ground_truth_path = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak/voc2012/unet_plots_full_2_correct/ground_truth.pt"
#ground_truth_path = "/work/ws-tmp/sa058646-segment/semseg/runs/new_exp_slak_fgsm_testing2/voc2012/pspnet_plots_full_2_correct2/ground_truth.pt"
ground_truth = torch.load(ground_truth_path)
gt_kvals = ground_truth[0].cpu().numpy()
gt_Abins = ground_truth[1].cpu().numpy()
all_tensors = glob.glob(parent_dir)
gt_Abins = gt_Abins/gt_Abins[0]
#import ipdb;ipdb.set_trace()

for tensor in all_tensors:
    if 'convnext' in tensor or True:
        points = torch.load(tensor)
        kvals = points[0].cpu().numpy()
        Abins = points[1].cpu().numpy()
        name = os.path.basename(tensor).replace('.pt','')
        name = os.path.basename(name).replace('convnext_tiny_','')
        #import ipdb;ipdb.set_trace()
        if 'trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0' in name:
            plt.loglog(kvals, Abins, label=name, color='k')
        elif "False" in name:
            plt.loglog(kvals, Abins, label=name, linestyle='-.', color=colors[count])
        elif "trans_kernel_31_small_trans_3" in name:
            plt.loglog(kvals, Abins, label=name, color=colors[count])
        elif "trans_kernel_31" in name:
            plt.loglog(kvals, Abins, label=name, linestyle='dotted', color=colors[count])
        elif "trans_kernel_2" in name:
            plt.loglog(kvals, Abins, label=name, linestyle='--', color=colors[count])
        else:
            plt.loglog(kvals, Abins, label=name, color=colors[count])
        count +=1
plt.loglog(gt_kvals, gt_Abins, label='ground_truth', color='k', linestyle='-.')
plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.title("convnext_tiny freq upsampling")
#plt.title("pspnet resnet50")
plt.legend(bbox_to_anchor=(1.04, 1), loc="best")
#plt.tight_layout()
#
#x1, x2 = plt.xlim()
#plt.xlim(100, x2)
y1, y2 = plt.ylim()
x1, x2 = plt.xlim()
#plt.ylim(y1, 10**7)
#plt.xlim(20,x2)
#import ipdb;ipdb.set_trace()
plt.savefig(save_path, dpi = 500, bbox_inches = "tight")








"""
curr_dir = os.getcwd()
img = cv2.imread(curr_dir+'/temp.png',0)
print( img.shape )

# Fourier Transform along the first axis

# Round up the size along this axis to an even number
n = int( math.ceil(img.shape[0] / 2.) * 2 )

# We use rfft since we are processing real values
a = np.fft.rfft(img,n, axis=0)

# Sum power along the second axis
a = a.real*a.real + a.imag*a.imag
a = a.sum(axis=1)/a.shape[1]

# Generate a list of frequencies
f = np.fft.rfftfreq(n)

# Graph it
plt.plot(f[1:],a[1:], label = 'sum of amplitudes over y vs f_x')

# Fourier Transform along the second axis

# Same steps as above
n = int( math.ceil(img.shape[1] / 2.) * 2 )

a = np.fft.rfft(img,n,axis=1)

a = a.real*a.real + a.imag*a.imag
a = a.sum(axis=0)/a.shape[0]

f = np.fft.rfftfreq(n)

plt.plot(f[1:],a[1:],  label ='sum of amplitudes over x vs f_y')

plt.ylabel( 'amplitude' )
plt.xlabel( 'frequency' )
plt.yscale( 'log' )

plt.legend()

plt.savefig( 'test_rfft.png' , transparent=True )
#plt.show()



        mag_path = os.path.join(freq_folder, image_name + '_mag.png')
        freq_path = os.path.join(freq_folder, image_name + '_freq.png')
        post_path = os.path.join(freq_folder, image_name + '_map.png')
        new_col, new_row, rows, cols=True, True, 0,0
        for it in range(posterior.shape[-1]):
            maps=posterior[:,:,it]
            if new_col:
                cols=maps
                new_col=False
            else:                
                cols=np.concatenate((cols,maps), axis=1)
            if (it+1)%7==0:
                cols.shape
                if new_row:
                    rows=cols
                    new_row=False
                else:
                    rows=np.vstack([rows, cols])
                new_col=True
        #import ipdb;ipdb.set_trace()
        #for i,j in zip(range(rows.shape[0]), range(rows.shape[1])):            
        #    if rows[i,j] < 1e-7:
        #        rows[i,j] = 255
            #    item=0
        #import ipdb;ipdb.set_trace()        
        (thresh, blackAndWhiteImage) = cv2.threshold(rows, 0.1, 255, cv2.THRESH_BINARY)        
        cv2.imwrite(post_path, blackAndWhiteImage)
        cv2.imwrite(mag_path, magnitude_specturm)
        cv2.imwrite(freq_path, img_back)
        gray_posterior= np.uint8(posterior)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_specturm = np.abs(fshift)
        row, col = gray.shape
        win_row, win_col = row//2, col//2        
        fshift[win_row-30:win_row+30, win_col-30:win_col+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)


"""