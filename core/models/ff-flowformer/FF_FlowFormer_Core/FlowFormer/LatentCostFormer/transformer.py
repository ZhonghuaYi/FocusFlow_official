import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import cv2 as cv

from einops.layers.torch import Rearrange
from einops import rearrange

from ..utils import coords_grid, bilinear_sampler, upflow8
from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from ..encoders import twins_svt_large_context, twins_svt_large, twins_svt_large_CCE
from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
from .twins import PosConv
from .encoder import MemoryEncoder, Fusion_MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder


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
    if cfg.MASK_MODAL == 'context':
        dilate = cfg.MASK_DILATE
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate))
        kernel = torch.FloatTensor(kernel).to(mask1.device)[None, None, :, :]
        mask1 = F.conv2d(mask1/255, kernel, padding=dilate//2) > 0
        mask1 = (mask1) * image1
        mask2 = image2.clone()
    elif cfg.MASK_MODAL == 'point':
        mask_channel = cfg.MASK_CHANNEL
        B, C, H, W = mask1.shape
        assert C == 1
        if mask_channel != C:
            mask1 = mask1.repeat(1, mask_channel, 1, 1)
        
        mask2 = torch.ones_like(mask1, device=image2.device) * 255
            
    elif cfg.MASK_MODAL == 'neighborE':
        mask_channel = cfg.MASK_CHANNEL
        B, C, H, W = mask1.shape
        assert C == 1
        dilate = cfg.MASK_DILATE
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate))
        kernel = torch.FloatTensor(kernel).to(mask1.device)[None, None, :, :]
        mask1 = F.conv2d(mask1/255, kernel, padding=dilate//2) > 0
        mask1 = mask1 * 255
        if mask_channel != C:
            mask1 = mask1.repeat(1, mask_channel, 1, 1)
            
        mask2 = torch.ones_like(mask1, device=image2.device) * 255
        
    elif cfg.MASK_MODAL == 'neighborG':
        mask_channel = cfg.MASK_CHANNEL
        B, C, H, W = mask1.shape
        assert C == 1
        kernel = get_kernel(cfg.KERNEL_SIZE, cfg.KERNEL_SIGMA)
        kernel = torch.FloatTensor(kernel).to(mask1.device)
        padding = cfg.KERNEL_SIZE//2
        mask1 = F.conv2d(mask1, kernel, padding=padding)
        mask1 = mask1 * 255 / mask1.max()
        if mask_channel != C:
            mask1 = mask1.repeat(1, mask_channel, 1, 1)
            
        mask2 = torch.ones_like(mask1, device=image2.device) * 255
        
    elif cfg.MASK_MODAL == 'frame':
        mask1 = image1.clone()
        mask2 = image2.clone()
        
    return mask1, mask2


class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        # if cfg.cnet == 'twins':
        #     self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        # elif cfg.cnet == 'basicencoder':
        #     self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        
        if cfg.pretrain_model is not None:
            model_dict = torch.load(cfg.pretrain_model)
            if not hasattr(self, 'module'):
                model_dict = {key.replace('module.', ''): value for key, value in model_dict.items()}
            self.load_state_dict(model_dict, strict=True)
            print("Load pretrained model from {}".format(cfg.pretrain_model))


    def forward(self, image1, image2, mask1=None, mask2=None, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        return flow_predictions
    

class FF_FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FF_FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = Fusion_MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        # if cfg.cnet == 'twins':
        #     self.context_encoder = twins_svt_large_CCE()
        # elif cfg.cnet == 'basicencoder':
        #     self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        self.context_encoder = twins_svt_large_CCE()
        
        if cfg.pretrain_model is not None:
            model_dict = torch.load(cfg.pretrain_model)
            if not hasattr(self, 'module'):
                model_dict = {key.replace('module.', ''): value for key, value in model_dict.items()}
            self.load_state_dict(model_dict, strict=True)
            print("Load pretrained model from {}".format(cfg.pretrain_model))
            
        if cfg.load_former is not None:
            model_dict = torch.load(cfg.load_former)
            if not hasattr(self, 'module'):
                model_dict = {key.replace('module.', ''): value for key, value in model_dict.items()}
            self.load_state_dict(model_dict, strict=False)
            print("Load pretrained FlowFormer part from {}".format(cfg.load_former))


    def forward(self, image1, image2, mask1, mask2, output=None, flow_init=None):
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        mask1, mask2 = init_mask(image1, image2, mask1, mask2, self.cfg)
        
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        mask1 = 2 * (mask1 / 255.0) - 1.0
        mask2 = 2 * (mask2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1, mask1)
            
        cost_memory = self.memory_encoder(image1, image2, mask1, mask2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        return flow_predictions
