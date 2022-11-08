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

cv2.ocl.setUseOpenCL(False)
maps = 0
outputs = []
image_names = []
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
    check_makedirs(feature_folder)

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

    args.trans_kernel = [2, 2, 2] if not hasattr(args, 'trans_kernel') else [int(args.trans_kernel), int(args.trans_kernel), int(args.trans_kernel)]
    args.backbone_kernel = [7, 7, 7] if not hasattr(args, 'backbone_kernel') else [int(args.backbone_kernel), int(args.backbone_kernel), int(args.backbone_kernel)]
    args.use_convnext_backbone = False if not hasattr(args, 'use_convnext_backbone') else args.use_convnext_backbone
    args.small_trans = 0 if not hasattr(args, 'small_trans') else int(args.small_trans)
    args.small_conv = 0 if not hasattr(args, 'small_conv') else int(args.small_conv)
    
    if not args.has_prediction:
        if args.arch == 'psp':
            from model.pspnet_convnext import PSPNet
            model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        elif args.arch == 'psa':
            from model.psanet import PSANet
            model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, compact=args.compact,
                           shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                           normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, pretrained=False)
        elif args.arch == 'unet':
            backbones = ['resnet18','resnet34','resnet50','resnet101','resnet152']
            convnext_backbones = ['convnext_tiny', 'convnext_small', 'convnext_base', 
                                'convnext_large', 'convnext_xlarge']
            slak_backbones = ['SLaK_tiny', 'SLaK_small', 'SLaK_base']
            if args.backbone in backbones:
                from model.unet import UNetResnet
                model = UNetResnet(num_classes=args.classes,
                                in_channels=3, backbone=args.backbone,
                                pretrained=args.pretrained, 
                                trans_kernel=args.trans_kernel, 
                                backbone_kernel=args.backbone_kernel, 
                                use_convnext_backbone=args.use_convnext_backbone,
                                small_trans=args.small_trans,
                                small_conv=args.small_conv)
            elif args.backbone in convnext_backbones:
                from model.unet import UNetConvNeXt
                model = UNetConvNeXt(num_classes=args.classes,
                                in_channels=3, backbone=args.backbone,
                                pretrained=args.pretrained,
                                trans_kernel=args.trans_kernel, 
                                backbone_kernel=args.backbone_kernel, 
                                use_convnext_backbone=args.use_convnext_backbone,
                                small_trans=args.small_trans,
                                small_conv=args.small_conv)
            elif args.backbone in slak_backbones:
                from model.unet import UNetSLaK
                model = UNetSLaK(num_classes=args.classes,
                                in_channels=3, backbone=args.backbone,
                                pretrained=args.pretrained,
                                kernel_size=args.kernel_size,
                                width_factor=args.width_factor, Decom= args.Decom, 
                                trans_kernel=args.trans_kernel, 
                                backbone_kernel=args.backbone_kernel, 
                                use_convnext_backbone=args.use_convnext_backbone,
                                small_trans=args.small_trans,
                                small_conv=args.small_conv)
            else:
                from model.unet import UNet
                model = UNet(num_classes=args.classes, in_dim=3, conv_dim=64)
        elif args.arch =='unet2':
            from model.unet2 import UNet
            model = UNet(num_classes=args.classes, in_dim=3, conv_dim=64)
        else:
            raise Exception('{} architecture not supported yet'.format(args.arch))

        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        test(test_loader, test_data.data_list, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, freq_folder, colors, feature_folder=feature_folder)
    if args.split != 'test':
        cal_acc(test_data.data_list, gray_folder, args.classes, names)

def plot_features(maps, image_name=None, feature_folder=None):
    horizontal = 0
    verticle = 0
    first_h = True
    first_v = True
    index = 0
    #import ipdb;ipdb.set_trace()
    #for _ in range(12):
    #    for _ in range(8):
    #        ax = plt.subplot(12, 8, ix)
    #        ax.set_xticks([])
    #        ax.set_yticks([])
    #        plt.imshow(map[ix-1,:,:])
    #        ix += 1
    #maps_name = feature_folder + '/' + image_name+'_output_map.png'
    maps_name = feature_folder + '/' + image_name+'_binary_map.png'
    #plt.savefig(maps_name, dpi=500)
    for index in range(96):
        feat = np.uint8(maps[index])
        feat[feat<125]=0
        feat[feat>125]=1
        #import ipdb;ipdb.set_trace()
        if first_h:
            horizontal = feat
            first_h = False
        else:
            horizontal=np.hstack((horizontal, feat))
        if (index+1)%12==0:
            first_h=True
            if first_v:
                verticle = horizontal                
                first_v=False
            else:
                verticle = np.vstack((verticle, horizontal))
    #import ipdb;ipdb.set_trace()
    cv2.imwrite(maps_name, verticle)

def plot_posterior(maps, image_name=None, feature_folder=None):
    horizontal = 0
    verticle = 0
    first_h = True
    first_v = True
    index = 0
    #import ipdb;ipdb.set_trace()
    #for _ in range(12):
    #    for _ in range(8):
    #        ax = plt.subplot(12, 8, ix)
    #        ax.set_xticks([])
    #        ax.set_yticks([])
    #        plt.imshow(map[ix-1,:,:])
    #        ix += 1
    maps_name = feature_folder + '/' + image_name+'_output_map.png'
    
    #plt.savefig(maps_name, dpi=500)
    for index in range(21):
        feat = np.uint8(maps[index])
        #import ipdb;ipdb.set_trace()
        if first_h:
            horizontal = feat
            first_h = False
        else:
            horizontal=np.hstack((horizontal, feat))
        if (index+1)%7==0:
            first_h=True
            if first_v:
                verticle = horizontal                
                first_v=False
            else:
                verticle = np.vstack((verticle, horizontal))
    import ipdb;ipdb.set_trace()
    cv2.imwrite(maps_name, verticle)
    

    


def net_process(model, image, mean, std=None, flip=False, image_name=None, feature_folder=None):    
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    global maps, output_before_softmax, args, outputs
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        if args.arch=='unet':
            import ipdb;ipdb.set_trace()
            maps = model(input)            
            output = model.module.last.out(maps)            
            maps = maps.detach().cpu()#.numpy()
        else:
            output = model(input)
    #plot_posterior(output[0].detach().cpu(), image_name, feature_folder)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    #import ipdb;ipdb.set_trace()
    maps = maps[0]
    #plot_features(maps, image_name, feature_folder)
    #maps = model.module.feature_map
    #output_before_softmax = torch.clone(output)[0]
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    input.detach().cpu()
    del input
    torch.cuda.empty_cache()
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3, image_name=None, feature_folder=None):
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
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std, image_name=image_name, feature_folder=feature_folder)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def power_spectra(data, freq_path):    
    global args
    d2 = torch.device('cuda:1')
    first = True
    first_amp = True
    count = 0
    amplitudes = []
    for split_data in torch.split(torch.stack(data), 1):
        for img in split_data[0]:            
            if args.binarize:
                img = img.cpu().numpy()
                img = np.uint8(img)
                img[img<128] = 0
                img[img>128] = 1
                img = torch.tensor(img)
            if first:
                freq_domain = np.expand_dims(torch.fft.fftn(img.to(d2)).detach().cpu().numpy(), axis=0)
                first = False      
            else:
                freq_domain = np.vstack((freq_domain, np.expand_dims(torch.fft.fftn(img.to(d2)).detach().cpu().numpy(), axis=0)))
        if first_amp:
            amplitudes = np.expand_dims(np.average(np.abs(freq_domain)**2, axis=0), axis =0)
            first_amp = False
        else:
            amplitudes = np.vstack((amplitudes, np.expand_dims(np.average(np.abs(freq_domain)**2, axis=0), axis =0)))
        first=True
    
    amplitude = np.average(np.asarray(amplitudes), axis=0)
   
    npix = amplitude.shape[0]
    kfreq = np.fft.fftfreq(npix) *npix

    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = amplitude.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    save_name = Path(freq_path).parent.parent.parent.name + '.pt'
    file_name = os.path.join(freq_path, save_name)
    
    torch.save(torch.tensor([kvals, Abins], device='cpu'), file_name)
    """
    plt.loglog(kvals, Abins)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.tight_layout()
    plt.savefig("cloud_power_spectrum2.png", dpi = 300, bbox_inches = "tight")
    """


def test(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, freq_folder, colors, feature_folder=None):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    global maps, output_before_softmax
    all_maps = []    
    
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std, image_name=image_name, feature_folder=feature_folder)
        prediction /= len(scales)
        posterior = prediction
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        check_makedirs(freq_folder)
        gray = np.uint8(prediction)
        all_maps.append(maps)#.transpose(1,2,0))
        
        color = colorize(gray, colors)
        #image_path, _ = data_list[i]
        #image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')

        cv2.imwrite(gray_path, gray)

        
        color.save(color_path)
    #power_spectra(all_maps, freq_folder)
    

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    global image_names

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        image_names.append(image_name)
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))


if __name__ == '__main__':
    main()
