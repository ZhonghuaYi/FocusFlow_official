import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv

from .parallel_fusion import FusionUnit

try:
    from correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, '../correlation'); import correlation
    print("load from ../correlation")
    # you should consider upgrading python
# end


backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask
# end


def get_kernel(kernel_size, sigma):
    sigma_3 = 3 * sigma
    X = np.linspace(-sigma_3, sigma_3, kernel_size)
    Y = np.linspace(-sigma_3, sigma_3, kernel_size)
    x, y = np.meshgrid(X, Y)
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
    Z = gauss.sum()
    kernel = (1/Z)*gauss
    return torch.FloatTensor(kernel).view(1, 1, kernel_size, kernel_size)


def init_mask(image1, image2, mask1, mask2, cfg):
    if cfg.TRAIN.MASK_MODAL == 'context':
        dilate = cfg.TRAIN.MASK_DILATE
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate))
        kernel = torch.FloatTensor(kernel).to(mask1.device)[None, None, :, :]
        mask1 = F.conv2d(mask1/255, kernel, padding=dilate//2) > 0
        mask1 = (mask1) * image1
        mask2 = image2.clone()
    elif cfg.TRAIN.MASK_MODAL == 'point':
        mask_channel = cfg.TRAIN.MASK_CHANNEL
        B, C, H, W = mask1.shape
        assert C == 1
        if mask_channel != C:
            mask1 = mask1.repeat(1, mask_channel, 1, 1)
        
        mask2 = torch.ones_like(mask1, device=image2.device) * 255
            
    elif cfg.TRAIN.MASK_MODAL == 'neighborE':
        mask_channel = cfg.TRAIN.MASK_CHANNEL
        B, C, H, W = mask1.shape
        assert C == 1
        dilate = cfg.TRAIN.MASK_DILATE
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate))
        kernel = torch.FloatTensor(kernel).to(mask1.device)[None, None, :, :]
        mask1 = F.conv2d(mask1/255, kernel, padding=dilate//2) > 0
        mask1 = mask1 * 255
        if mask_channel != C:
            mask1 = mask1.repeat(1, mask_channel, 1, 1)
            
        mask2 = torch.ones_like(mask1, device=image2.device) * 255
        
    elif cfg.TRAIN.MASK_MODAL == 'neighborG':
        mask_channel = cfg.TRAIN.MASK_CHANNEL
        B, C, H, W = mask1.shape
        assert C == 1
        kernel = get_kernel(cfg.TRAIN.KERNEL_SIZE, cfg.TRAIN.KERNEL_SIGMA)
        kernel = torch.FloatTensor(kernel).to(mask1.device)
        padding = cfg.TRAIN.KERNEL_SIZE//2
        mask1 = F.conv2d(mask1, kernel, padding=padding)
        mask1 = mask1 * 255 / mask1.max()
        if mask_channel != C:
            mask1 = mask1.repeat(1, mask_channel, 1, 1)
            
        mask2 = torch.ones_like(mask1, device=image2.device) * 255
        
    elif cfg.TRAIN.MASK_MODAL == 'frame':
        mask1 = image1.clone()
        mask2 = image2.clone()
        
    return mask1, mask2


class FF_PWCNET(torch.nn.Module):
    def __init__(self, cfg, pretrain=None, load_pwcnet=None):
        super().__init__()
        if cfg.MODEL.FUSION != 'parallel':
            raise NotImplementedError(f"FF_PWCNET only support parallel fusion, but got {cfg.MODEL.FUSION}")
        
        self.fusion_type = cfg.MODEL.FUSION_TYPE
        self.cfg = cfg
        print(f"Parallel fusion type: {self.fusion_type}")

        class Extractor(torch.nn.Module):
            def __init__(self, fusion_type):
                super().__init__()
                self.fusion_type = fusion_type

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.mask_netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.fusion1 = FusionUnit(16, self.fusion_type, True)

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.mask_netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.fusion2 = FusionUnit(32, self.fusion_type, True)

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.mask_netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.fusion3 = FusionUnit(64, self.fusion_type, True)

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.mask_netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.fusion4 = FusionUnit(96, self.fusion_type, True)

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.mask_netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.fusion5 = FusionUnit(128, self.fusion_type, True)

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.mask_netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.fusion6 = FusionUnit(196, self.fusion_type, False)
            # end

            def forward(self, tenInput, mask):
                tenOne = self.netOne(tenInput)
                mask = self.mask_netOne(mask)
                mask, tenOne = self.fusion1(mask, tenOne)
                
                tenTwo = self.netTwo(tenOne)
                mask = self.mask_netTwo(mask)
                mask, tenTwo = self.fusion2(mask, tenTwo)
                
                tenThr = self.netThr(tenTwo)
                mask = self.mask_netThr(mask)
                mask, tenThr = self.fusion3(mask, tenThr)
                
                tenFou = self.netFou(tenThr)
                mask = self.mask_netFou(mask)
                mask, tenFou = self.fusion4(mask, tenFou)
                
                tenFiv = self.netFiv(tenFou)
                mask = self.mask_netFiv(mask)
                mask, tenFiv = self.fusion5(mask, tenFiv)
                
                tenSix = self.netSix(tenFiv)
                mask = self.mask_netSix(mask)
                mask, tenSix = self.fusion6(mask, tenSix)
                
                self.fmap = tenFou.detach().clone()

                return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end

                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }
            # end
        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        self.netExtractor = Extractor(self.fusion_type)

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        if pretrain is not None:
            self.load_state_dict(torch.load(pretrain), strict=True)
            print('load pretrain model from {}'.format(pretrain))
        
        if load_pwcnet is not None:
            self.load_state_dict(torch.load(load_pwcnet), strict=False)
            print('load pwcnet from {}'.format(load_pwcnet))

    def preprocess(self, tenOne, tenTwo, mask1, mask2):
        _, _, H, W = tenOne.shape
        self.origin_H = H
        self.origin_W = W
        new_H = int(math.floor(math.ceil(H / 64.0) * 64.0))
        new_W = int(math.floor(math.ceil(W / 64.0) * 64.0))
        self.new_H = new_H
        self.new_W = new_W
        tenOne = torch.nn.functional.interpolate(input=tenOne, size=(new_H, new_W), mode='bilinear', align_corners=False)
        tenTwo = torch.nn.functional.interpolate(input=tenTwo, size=(new_H, new_W), mode='bilinear', align_corners=False)
        mask1 = torch.nn.functional.interpolate(input=mask1, size=(new_H, new_W), mode='bilinear', align_corners=False)
        mask2 = torch.nn.functional.interpolate(input=mask2, size=(new_H, new_W), mode='bilinear', align_corners=False)
        return tenOne, tenTwo, mask1, mask2


    def forward(self, tenOne, tenTwo, mask1, mask2, test_mode=False):
        tenOne, tenTwo, mask1, mask2 = self.preprocess(tenOne, tenTwo, mask1, mask2)
        
        mask1, mask2 = init_mask(tenOne, tenTwo, mask1, mask2, self.cfg)
        
        tenOne = self.netExtractor(tenOne, mask1)
        tenTwo = self.netExtractor(tenTwo, mask2)

        flow_list = []

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        flow_list.insert(0, objEstimate['tenFlow'])
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        flow_list.insert(0, objEstimate['tenFlow'])
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        flow_list.insert(0, objEstimate['tenFlow'])
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        flow_list.insert(0, objEstimate['tenFlow'])
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)
        objEstimate['tenFlow'] = objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])
        flow_list.insert(0, objEstimate['tenFlow'])

        if test_mode:
            out = torch.nn.functional.interpolate(input=objEstimate['tenFlow'], size=(self.origin_H, self.origin_W), mode='bilinear', align_corners=False)
            out[:, 0, :, :] = out[:, 0, :, :] * self.origin_W / self.new_W
            out[:, 1, :, :] = out[:, 1, :, :] * self.origin_H / self.new_H
            return out

        return flow_list
    # end
# end
        
    
