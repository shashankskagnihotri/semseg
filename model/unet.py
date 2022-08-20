from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
from model import resnet_3
from model import resnet_2
from model import inverse_resnet
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from model import convnext, SLaK
import torchvision.models
from collections import OrderedDict
import math



def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

def backbone_conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels*2, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

def backbone_convnext(in_channels, out_channels, kernel_size=7, inner_channels=None, small_conv=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels*2, inner_channels, kernel_size=kernel_size, 
                padding=(kernel_size-1)//2, bias=False, 
                groups=math.gcd(in_channels*2, inner_channels)) \
                if small_conv==0 else convolution(in_channels, out_channels, kernel_size, inner_channels, small_conv),
        permutation_one(),
        convnext.LayerNorm(inner_channels, eps=1e-6),
        nn.Linear(inner_channels, inner_channels*4),
        nn.GELU(),
        nn.Linear(inner_channels*4, out_channels),
        permutation_two()
        )
    return down_conv

def convTrans2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0):
    """convtranspose 2d"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, 
                                stride=stride, padding=padding, bias=False)

class permutation_one(nn.Module):
    def __init__(self):
        super(permutation_one, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

class permutation_two(nn.Module):
    def __init__(self):
        super(permutation_two, self).__init__()
    def forward(self, x):
        return x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

class convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, inner_channels, small_conv):
        super(convolution, self).__init__()
        self.large_convolution = nn.Conv2d(in_channels*2, inner_channels, kernel_size=kernel_size, 
                                    padding=(kernel_size-1)//2, bias=False, 
                                    groups=math.gcd(in_channels*2, inner_channels))        
        self.small_convolution = nn.Conv2d(in_channels*2, inner_channels, kernel_size=small_conv, 
                                    padding=(small_conv-1)//2, bias=False, 
                                    groups=math.gcd(in_channels*2, inner_channels))
    def forward(self, x):
        return self.large_convolution(x) + self.small_convolution(x)        

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class decoder_resnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, 
                backbone_kernel=7, use_convnext_backbone=False, 
                small_trans=None, small_conv=None):
        super(decoder_resnet, self).__init__()
        pos1 = int(in_channels*2)
        pos2 = int(in_channels)
        padding = 0 
        output_padding = 0 if kernel_size==2 else 1  
        if kernel_size != 2:
            padding = (kernel_size-1)//2

        #groups=pos2 if (kernel_size>2 and pos1%pos2==0) else 1         
        groups=math.gcd(pos1, pos2) if kernel_size>2 else 1 
        self.up = nn.ConvTranspose2d(pos1, pos2, kernel_size=kernel_size, 
                                    stride=2, padding=padding, groups=groups, 
                                    output_padding=output_padding)
        self.up_conv = backbone_convnext(int(in_channels), out_channels, 
                                            kernel_size=backbone_kernel, small_conv=small_conv) \
                        if use_convnext_backbone else backbone_conv(int(in_channels), out_channels)
        self.small_trans_kernel = nn.ConvTranspose2d(pos1, pos2, kernel_size=small_trans, 
                                    stride=2, padding=(small_trans-1)//2, groups=groups, 
                                    output_padding=output_padding) if small_trans!=0 else None

    def forward(self, x_copy, x, interpolate=True):       
        # Concatenate
        #x1 = self.up(x)
        #x = (x1 + self.small_trans_kernel(x)) if self.small_trans_kernel != None else x
        #x = self.up(x) + (self.small_trans_kernel(x) if self.small_trans_kernel != None else None)
        if self.small_trans_kernel != None:
            x = self.up(x) + self.small_trans_kernel(x)
        else:
            x = self.up(x)
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)        
        return x


class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 freeze_bn=False, **_):
        super(UNet, self).__init__()

        self.criterion=criterion
        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()




"""
-> Unet with a resnet backbone
"""

class UNetResnet(BaseModel):    
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', 
                pretrained=True, criterion=nn.CrossEntropyLoss(ignore_index=255), 
                freeze_bn=False, trans_kernel=[2, 2, 2], backbone_kernel=[7, 7, 7], 
                use_convnext_backbone=False, small_trans=None, small_conv=None, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        """
        #############    AWAITING UPGRADE     #############

        from torchvision.prototype import models as PM
        if backbone == 'resnet18':
            weights=PM.ResNet18_Weights.DEFAULT
        elif backbone == 'resnet34':
            weights=PM.ResNet34_Weights.DEFAULT
        elif backbone == 'resnet101':
            weights=PM.ResNet101_Weights.DEFAULT
        elif backbone == 'resnet152':
            weights=PM.ResNet152_Weights.DEFAULT
        else:
            weights=PM.ResNet50_Weights.DEFAULT
        model = getattr(PM, backbone)(weights=weights)
        """
        model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        base_width = model.fc.in_features
        global decoder
        #decoder_model = getattr(inverse_resnet, backbone)(pretrained=False, 
        #                        width_per_group = base_width)
        #model = getattr(resnet_2, backbone)(pretrained, norm_layer=nn.BatchNorm2d)

        self.criterion = criterion
        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        #self.middle = x2conv(base_width, base_width)

        # decoder   
        #self.decoder1 = decoder(int(base_width), int(base_width/2))
        self.decoder1 = x2conv(int(base_width), int(base_width))
        #self.decoder1 = decoder(1024, 512)
        self.decoder2 = decoder_resnet(int(base_width/2), int(base_width/2), 
                                        kernel_size=trans_kernel[0], small_trans=small_trans, 
                                        backbone_kernel=backbone_kernel[0], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        self.decoder3 = decoder_resnet(int(base_width/4), int(base_width/4), 
                                        kernel_size=trans_kernel[1], small_trans=small_trans, 
                                        backbone_kernel=backbone_kernel[1], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        self.decoder4 = decoder_resnet(int(base_width/8), int(base_width/8), 
                                        kernel_size=trans_kernel[2], small_trans=small_trans, 
                                        backbone_kernel=backbone_kernel[2], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        #self.decoder4 = decoder(int(base_width/16), int(base_width/32))
        self.last = nn.Sequential(OrderedDict([("up", nn.ConvTranspose2d(int(base_width/8), 
                                            int(base_width/32), 
                                            kernel_size=1, stride=1)),
                                ("conv", nn.Conv2d(int(base_width/16), 
                                int(base_width/32), kernel_size=1)),
                                ("norm", nn.ReLU(nn.BatchNorm2d(int(base_width/32)))),
                                ("lastup", nn.ConvTranspose2d(int(base_width/32),
                                    int(base_width/32), kernel_size=4, stride=4)),
                                ("out", nn.Conv2d(int(base_width/32), 
                                num_classes, kernel_size=1))]))
        """
        self.decoder1 = decoder_model.layer1
        self.upconv1 = convTrans2d(int(base_width/4), int(base_width/2), kernel_size=2, stride=2)
        self.decoder2 = decoder_model.layer2
        self.upconv2 = convTrans2d(int(base_width/8), int(base_width/4), kernel_size=4, stride=4)
        self.decoder3 = decoder_model.layer3
        self.upconv3 = convTrans2d(int(base_width/16), int(base_width/8), kernel_size=4, stride=4)
        self.decoder4 = decoder_model.layer4
        self.upconv4 = convTrans2d(int(base_width/32), int(base_width/32), kernel_size=2, stride=2)
        self.last = nn.Sequential(nn.Conv2d(int(base_width/16), int(base_width/32), kernel_size=1, 
                                            stride=1, padding=0),
                                nn.BatchNorm2d(int(base_width/32)), nn.ReLU(inplace=True),                                
                                convTrans2d(int(base_width/32), 
                                            int(base_width/32), 
                                            #num_classes,
                                            kernel_size=4, stride=4),
                                nn.Conv2d(int(base_width/32), num_classes, kernel_size=1, 
                                        stride=1, padding=0))
        """

        if pretrained:
            print(":::::::::>>>>>>>       INITIALIZING ONLY THE DECODER")
            decoder = [self.decoder1, self.decoder2, self.decoder3,
                        self.decoder4, self.last]#, self.middle]
            for dec in decoder:
                initialize_weights(dec)
        else:
            print(":::::::::>>>>>>>       INITIALIZING THE WHOLE ARCHITECTURE")
            initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        # encoding
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #x5 = self.middle(x4)

        #decoding
        #import ipdb;ipdb.set_trace()
        x = self.decoder1(x4)        
        #x = self.upconv1(x)
        
        #x = torch.cat([x3, x], dim=1)

        x = self.decoder2(x3, x)
        #x = self.upconv2(x)
        
        #import ipdb;ipdb.set_trace()
        #x = torch.cat([x2, x], dim=1)

        x = self.decoder3(x2, x)
        #x = self.upconv3(x)
        
        #x = torch.cat([x1, x], dim=1)

        x = self.decoder4(x1, x)
        #x = self.upconv4(x)

        x = self.last.up(x)        
        x = torch.cat([x0, x], dim=1)
        x = self.last.norm(self.last.conv(x))
        x = self.last.lastup(x)
        x = self.last.out(x)

        """
        H, W = x.size(2), x.size(3)
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        x = self.conv7(self.conv6(x))
        """

        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                    self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(),
                    self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(),
                    self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


"""
-> Unet with a ConvNext backbone
    ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
"""

class UNetConvNeXt(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='convnext_tiny', 
                pretrained=True, criterion=nn.CrossEntropyLoss(ignore_index=255),
                freeze_bn=False, trans_kernel=[2, 2, 2], backbone_kernel=[7, 7, 7], 
                use_convnext_backbone=False, small_trans=None, small_conv=None, freeze_backbone=False,**_):
        super(UNetConvNeXt, self).__init__()
        
        self.criterion = criterion
        model = getattr(convnext, backbone)(pretrained=pretrained)
        base_width = model.head.in_features

        #self.initial = list(model.children())[:1]
        self.initial = list(model.downsample_layers[0])
        #TODO: get only the head here and modify it similar to resnet50
        
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4, padding=0, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.stages[0]
        self.layer2 = nn.Sequential(model.downsample_layers[1], model.stages[1])
        self.layer3 = nn.Sequential(model.downsample_layers[2], model.stages[2])
        self.layer4 = nn.Sequential(model.downsample_layers[3], model.stages[3])

        # decoder 
        """
        #TODO: modify it as per ConvNeXt dimensions
        """
        self.decoder1 = x2conv(int(base_width), int(base_width))
        #self.decoder1 = decoder(1024, 512)
        self.decoder2 = decoder_resnet(int(base_width/2), int(base_width/2), 
                                        kernel_size=trans_kernel[0], small_trans=small_trans,
                                        backbone_kernel=backbone_kernel[0], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        self.decoder3 = decoder_resnet(int(base_width/4), int(base_width/4), 
                                        kernel_size=trans_kernel[1], small_trans=small_trans, 
                                        backbone_kernel=backbone_kernel[1], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        self.decoder4 = decoder_resnet(int(base_width/8), int(base_width/8), 
                                        kernel_size=trans_kernel[2], small_trans=small_trans, 
                                        backbone_kernel=backbone_kernel[2], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        #self.decoder4 = decoder(int(base_width/16), int(base_width/32))
        self.last = nn.Sequential(OrderedDict([
                                ("conv", nn.Conv2d(int(base_width/4), 
                                int(base_width/8), kernel_size=1)),
                                ("norm", nn.ReLU(nn.BatchNorm2d(int(base_width/8)))),
                                ("lastup", nn.ConvTranspose2d(int(base_width/8),
                                    int(base_width/8), kernel_size=4, stride=4)),
                                ("out", nn.Conv2d(int(base_width/8), 
                                num_classes, kernel_size=1))]))
        """
                                ("up", nn.ConvTranspose2d(int(base_width/8), 
                                            int(base_width/32), 
                                            kernel_size=1, stride=1)),
        """

        if pretrained:
            print(":::::::::>>>>>>>       INITIALIZING ONLY THE DECODER")
            decoder = [self.decoder1, self.decoder2, self.decoder3,
                        self.decoder4, self.last]#, self.middle]
            for dec in decoder:
                initialize_weights(dec)
        else:
            print(":::::::::>>>>>>>       INITIALIZING THE WHOLE ARCHITECTURE")
            initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        # encoding
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #x5 = self.middle(x4)

        #decoding
        #import ipdb;ipdb.set_trace()
        x = self.decoder1(x4)        
        #x = self.upconv1(x)
        
        #x = torch.cat([x3, x], dim=1)

        x = self.decoder2(x3, x)
        #x = self.upconv2(x)
        
        #import ipdb;ipdb.set_trace()
        #x = torch.cat([x2, x], dim=1)

        x = self.decoder3(x2, x)
        #x = self.upconv3(x)
        
        #x = torch.cat([x1, x], dim=1)

        x = self.decoder4(x1, x)
        #x = self.upconv4(x)

        #x = self.last.up(x)        
        x = torch.cat([x0, x], dim=1)
        x = self.last.norm(self.last.conv(x))
        x = self.last.lastup(x)
        x = self.last.out(x)
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                    self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(),
                    self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(),
                    self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


"""
-> Unet with a SLaK backbone
    SLaK
        A PyTorch impl of : 
        More ConvNets in the 2020s: Scaling up Kernels Beyond 51 x 51 using Sparsity  -
          https://arxiv.org/abs/2207.03620
"""

class UNetSLaK(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='SLaK_tiny', 
                pretrained=True, criterion=nn.CrossEntropyLoss(ignore_index=255),
                drop_path_rate=0.0, kernel_size=[51,49,47,13,5], width_factor=1.3, Decom=True,
                bn=True, trans_kernel=[2, 2, 2], backbone_kernel=[7, 7, 7], small_conv=None, 
                use_convnext_backbone=False, small_trans=None, freeze_bn=False, freeze_backbone=False,**_):
        super(UNetSLaK, self).__init__()
        
        self.criterion = criterion
        model = getattr(SLaK, backbone)(drop_path_rate=drop_path_rate, kernel_size=kernel_size,
                                        width_factor=width_factor, Decom=Decom, bn=bn)
        if pretrained:
            if 'tiny' in backbone:
                checkpoint = torch.load("/work/ws-tmp/sa058646-segment/SLaK/models/SLaK_tiny_checkpoint.pth", map_location=torch.device('cpu'))['model']
            elif 'small' in backbone:
                checkpoint = torch.load("/work/ws-tmp/sa058646-segment/SLaK/models/SLaK_small_checkpoint.pth", map_location=torch.device('cpu'))['model']
            elif 'base' in backbone:
                checkpoint = torch.load("/work/ws-tmp/sa058646-segment/SLaK/models/SLaK_base_checkpoint.pth", map_location=torch.device('cpu'))['model']
            else:
                raise NotImplementedError('pretrained specified for an architecture that is not supported!')
            model.load_state_dict(checkpoint)
        base_width = model.head.in_features
        self.base_width=base_width        
        
        self.initial = list(model.downsample_layers[0])  
        
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4, padding=0, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.stages[0]
        self.layer2 = nn.Sequential(model.downsample_layers[1], model.stages[1])
        self.layer3 = nn.Sequential(model.downsample_layers[2], model.stages[2])
        self.layer4 = nn.Sequential(model.downsample_layers[3], model.stages[3])

        # decoder         
        self.decoder1 = x2conv(int(base_width), int(base_width))
        self.decoder2 = decoder_resnet(base_width/2, int(base_width/2), 
                                        kernel_size=trans_kernel[0], small_trans=small_trans,
                                        backbone_kernel=backbone_kernel[0], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        self.decoder3 = decoder_resnet(base_width/4, int(base_width/4), small_trans=small_trans,
                                        kernel_size=trans_kernel[1], backbone_kernel=backbone_kernel[1], 
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
        self.decoder4 = decoder_resnet(base_width/8, int(base_width/8), kernel_size=trans_kernel[2], 
                                        backbone_kernel=backbone_kernel[2], small_trans=small_trans,
                                        use_convnext_backbone=use_convnext_backbone,
                                        small_conv=small_conv)
    
        self.last = nn.Sequential(OrderedDict([
                                ("conv", nn.Conv2d(int(base_width/8)*2, 
                                int(base_width/8), kernel_size=1)),
                                ("norm", nn.ReLU(nn.BatchNorm2d(int(base_width/8)))),
                                ("lastup", nn.ConvTranspose2d(int(base_width/8),
                                    int(base_width/8), kernel_size=4, stride=4)),
                                ("out", nn.Conv2d(int(base_width/8), 
                                num_classes, kernel_size=1))]))
        """
                                ("up", nn.ConvTranspose2d(int(base_width/8), 
                                            int(base_width/32), 
                                            kernel_size=1, stride=1)),
        """

        if pretrained:
            print(":::::::::>>>>>>>       INITIALIZING ONLY THE DECODER")
            decoder = [self.decoder1, self.decoder2, self.decoder3,
                        self.decoder4, self.last]#, self.middle]
            for dec in decoder:
                initialize_weights(dec)
        else:
            print(":::::::::>>>>>>>       INITIALIZING THE WHOLE ARCHITECTURE")
            initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        # encoding
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #x5 = self.middle(x4)

        #decoding
        #import ipdb;ipdb.set_trace()
        x = self.decoder1(x4)        
        #x = self.upconv1(x)
        
        #x = torch.cat([x3, x], dim=1)

        x = self.decoder2(x3, x)
        #x = self.upconv2(x)
        
        #import ipdb;ipdb.set_trace()
        #x = torch.cat([x2, x], dim=1)

        x = self.decoder3(x2, x)
        #x = self.upconv3(x)
        
        #x = torch.cat([x1, x], dim=1)

        x = self.decoder4(x1, x)
        #x = self.upconv4(x)

        #x = self.last.up(x)        
        x = torch.cat([x0, x], dim=1)
        x = self.last.norm(self.last.conv(x))
        x = self.last.lastup(x)
        x = self.last.out(x)
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                    self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.decoder1.parameters(), self.decoder2.parameters(), self.decoder3.parameters(), 
                    self.decoder4.parameters(), self.last.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
