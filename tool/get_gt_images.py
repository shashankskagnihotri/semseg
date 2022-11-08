import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

import matplotlib.pyplot as plt
import os
import math
import matplotlib
import scipy.signal as signal
import scipy.stats as stats
from pathlib import Path

from scipy import fftpack
from tool import radialProfile

cv2.ocl.setUseOpenCL(False)
maps = 0
output_before_softmax = 0

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/voc2012/voc2012_unet_convnexttiny_250.yaml', help='config file')
    parser.add_argument('--testing', '-t', type=bool)
    parser.add_argument('opts', help='see config/voc2012/voc2012_unet_convnexttiny_250.yaml', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    elif args.arch == 'unet':
        print("::::::::::::::   Using UNet   ::::::::::::::")
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    global args, logger, maps, output_before_softmax
    args = get_parser()
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')
    freq_folder = os.path.join(args.save_folder, 'frequency')    
    feature_folder = os.path.join(args.save_folder, 'feature_maps')
    ground_truth = os.path.join(args.save_folder, 'ground_truth')
    check_makedirs(ground_truth)

    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    index_start = args.index_start
    #end = 500 if args.testing else len(test_data.data_list)
    end = len(test_data.data_list)
    if args.index_step == 0:
        index_end = end
        #index_end = 10
    else:
        index_end = min(index_start + args.index_step, end)
    if args.testing:
        test_data.data_list = test_data.data_list[index_start:index_end:2]
    else:
        test_data.data_list = test_data.data_list[index_start:index_end]
    #if args.testing:
    #    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, sampler = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=500))
    #else:
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]
    
    cal_acc(test_data.data_list, ground_truth, args.classes, names)

def cal_acc(data_list, freq_folder, classes, names):
    across_images = []
    first = True
    freq_folder = freq_folder.replace('_gpu/', '/')
    freq_folder = freq_folder.replace('/unet/', '/unet_11_binary_plots_full_2_correct/ground_truth/')
    from pathlib import Path
    freq_folder = Path(freq_folder)
    freq_folder = str(freq_folder.parent.parent.parent.parent)
    check_makedirs(freq_folder)
    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]        
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) 
        target *= 255
        save_name = freq_folder + '/' + image_name + '.png'        
        cv2.imwrite(save_name, target)





if __name__ == '__main__':
    main()



"""
 if first:
            #across_images=np.expand_dims(np.abs(F2)**2, axis=0)
            psd2D=np.abs(F2)**2
            psd1D=radialProfile.azimuthalAverage(psd2D)
            across_images=np.expand_dims(psd1D, axis=0)
            import ipdb;ipdb.set_trace()
        else:
            #across_images = np.vstack((across_images, np.expand_dims(np.abs(F2)**2, axis=0)), axis=0)
            psd2D=np.abs(F2)**2
            psd1D=radialProfile.azimuthalAverage(psd2D)
            across_images=np.vstack((np.expand_dims(psd1D, axis=0)))
            import ipdb;ipdb.set_trace()
"""