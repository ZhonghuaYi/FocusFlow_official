import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv

from .raft import RAFT
from .fusion import FusionLayer
from .conv_fusion import ConvFusionLayer


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


class FF_RAFT_FUSION(nn.Module):
    def __init__(self, pretrain=None, load_raft=None, use_fusion=None, fusion_channels=64, raft_small=False, dropout=0.,
                 alternate_corr=False, abandon_fnet=False, fuse_cnet=False, freeze_flownet=False, cfg=None):
        super().__init__()

        self.fusion_layer = None
        self.use_fusion = use_fusion
        self.freeze_flownet = freeze_flownet
        self.cfg = cfg
        if use_fusion is not None:
            if use_fusion == "attention":
                self.fusion_layer = FusionLayer(img_channel=3,
                                                mask_channel=3,
                                                wf=fusion_channels,
                                                depth=3,
                                                fuse_before_downsample=True,
                                                relu_slope=0.2,
                                                num_heads=[1, 2, 4])
                self.flow_net = RAFT(in_channels=fusion_channels, small=raft_small, dropout=dropout,
                                 alternate_corr=alternate_corr, abandon_fnet=abandon_fnet)
                if load_raft is not None:
                    self.flow_net.load_model(load_raft, flag="backend")
                    print("Load flow net backend.")
            elif use_fusion == "conv":
                self.fusion_layer = ConvFusionLayer(6, fusion_channels)
                self.flow_net = RAFT(in_channels=fusion_channels, small=raft_small, dropout=dropout,
                                 alternate_corr=alternate_corr, abandon_fnet=abandon_fnet)
                if load_raft is not None:
                    self.flow_net.load_model(load_raft, flag="backend")
                    print("Load flow net backend.")
                
            elif use_fusion == "parallel":

                self.flow_net = RAFT(in_channels=fusion_channels, small=raft_small, dropout=dropout,
                                    alternate_corr=alternate_corr, abandon_fnet=abandon_fnet,
                                    inside_fusion='parallel', fuse_cnet=fuse_cnet, cfg=cfg)
                
                if pretrain is not None:
                    self.load_state_dict(torch.load(pretrain), strict=True)
                    print("Load pretrained model from {}".format(pretrain))

                if load_raft is not None:
                    self.flow_net.load_model(load_raft, flag="all", strict=False)
                    print("Load all flow net.")
                
                if self.freeze_flownet:
                    self.freeze_self()
                    print("freeze flow net.")

        else:
            self.flow_net = RAFT(in_channels=3, small=raft_small, dropout=dropout, alternate_corr=alternate_corr)
            if pretrain is not None:
                    self.load_state_dict(torch.load(pretrain), strict=True)
                    print("Load pretrained model from {}".format(pretrain))

            if load_raft is not None:
                self.flow_net.load_model(load_raft, flag="all")
                print("Load all flow net.")

    def forward(self, image1, image2, mask1, mask2, raft_iters=12, flow_init=None, test_mode=False):

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        mask1, mask2 = init_mask(image1, image2, mask1, mask2, self.cfg)
        # mask2 = mask2 * image2
        
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        mask1 = 2 * (mask1 / 255.0) - 1.0
        mask2 = 2 * (mask2 / 255.0) - 1.0

        if self.fusion_layer is not None:
            f1 = self.fusion_layer(image1, mask1)
            f2 = self.fusion_layer(image2, mask2)

        else:
            f1 = image1
            f2 = image2

        if self.use_fusion == 'parallel':
            flow_predictions = self.flow_net(f1, f2, mask1, mask2, iters=raft_iters, flow_init=flow_init, test_mode=test_mode)
        else:
            flow_predictions = self.flow_net(f1, f2, iters=raft_iters, flow_init=flow_init, test_mode=test_mode)

        return flow_predictions
    
    def freeze_self(self):
        if self.use_fusion == 'parallel':
            self.flow_net.freeze_self(mode='parallel')
