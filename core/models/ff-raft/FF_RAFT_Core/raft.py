import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .parallel_fusion import BasicParallelFusionLayer
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

class ChannelProject(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        
        _, _, h, w = x.shape
        new_h, new_w = h//8, w//8
        x = nn.functional.interpolate(x, (new_h, new_w), mode="bilinear", align_corners=False)
        x = self.conv1(x)

        return x


class RAFT(nn.Module):
    def __init__(self, in_channels=3, small=False, dropout=0., alternate_corr=False, abandon_fnet=False, inside_fusion=None, fuse_cnet=False, cfg=None):
        super(RAFT, self).__init__()
        self.small = small
        self.abandon_fnet = abandon_fnet
        self.inside_fusion = inside_fusion
        self.fuse_cnet = fuse_cnet
        self.cfg = cfg

        if self.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4

        self.dropout = dropout

        self.alternate_corr = alternate_corr

        # feature network, context network, and update block
        if self.abandon_fnet:
            if self.small:
                self.channel_pjt = ChannelProject(in_channels, 128)
                self.cnet = SmallEncoder(in_channels, output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
                self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
            
            else:
                self.channel_pjt = ChannelProject(in_channels, 256)
                self.cnet = BasicEncoder(in_channels, output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
                self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
                
        else:
            if self.small:
                if self.inside_fusion is None:
                    self.fnet = SmallEncoder(in_channels, output_dim=128, norm_fn='instance', dropout=self.dropout)
                    self.cnet = SmallEncoder(in_channels, output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
                    self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
                elif self.inside_fusion == 'parallel':
                    self.fnet = BasicParallelFusionLayer(img_channel=3, mask_channel=cfg.TRAIN.MASK_CHANNEL, output_dim=128, norm_fn='instance', dropout=self.dropout, cfg=cfg)
                    if self.fuse_cnet:
                        self.cnet = BasicParallelFusionLayer(img_channel=3, mask_channel=cfg.TRAIN.MASK_CHANNEL, output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout, cfg=cfg)
                    else:
                        self.cnet = SmallEncoder(in_channel=3, output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
                    self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

            else:
                if self.inside_fusion is None:
                    self.fnet = BasicEncoder(in_channels, output_dim=256, norm_fn='instance', dropout=self.dropout)
                    self.cnet = BasicEncoder(in_channels, output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
                    self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)
                elif self.inside_fusion == 'parallel':
                    self.fnet = BasicParallelFusionLayer(img_channel=3, mask_channel=cfg.TRAIN.MASK_CHANNEL, output_dim=256, norm_fn='instance', dropout=self.dropout, cfg=cfg)
                    if self.fuse_cnet:
                        self.cnet = BasicParallelFusionLayer(img_channel=3, mask_channel=cfg.TRAIN.MASK_CHANNEL, output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout, cfg=cfg)
                    else:
                        self.cnet = BasicEncoder(in_channel=3, output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
                    self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
    def freeze_self(self, mode):
        if mode == 'parallel':
            self.fnet.freeze_self(mode)
            self.cnet.freeze_self(mode)
            self.update_block.freeze_self(mode)

    def load_model(self, model_path, flag="all", strict=True):
        if flag == "all":
            model_dict = torch.load(model_path)
            if not hasattr(self, 'module'):
                model_dict = {key.replace('module.', ''): value for key, value in model_dict.items()}

                self.load_state_dict(model_dict, strict=strict)

            else:
                self.module.load_state_dict(model_dict, strict=strict)

            if self.inside_fusion == 'parallel' and self.cfg.MODEL.LOAD_MODULE_TO_BRANCH:
                self.fnet.copy_to_branch()
                self.cnet.copy_to_branch()

        elif flag == "backend":
            model_dict = torch.load(model_path)
            if not hasattr(self, 'module'):
                model_dict = {key.replace('module.', ''): value for key, value in model_dict.items()}

                model_dict.pop("fnet.conv1.weight")
                model_dict.pop("fnet.conv1.bias")
                model_dict.pop("cnet.conv1.weight")
                model_dict.pop("cnet.conv1.bias")

                self.load_state_dict(model_dict, strict=False)

            else:
                model_dict.pop("module.fnet.conv1.weight")
                model_dict.pop("module.fnet.conv1.bias")
                model_dict.pop("module.cnet.conv1.weight")
                model_dict.pop("module.cnet.conv1.bias")

                self.module.load_state_dict(model_dict, strict=True)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, mask1=None, mask2=None, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        if self.abandon_fnet:
            fmap1 = self.channel_pjt(image1)
            fmap2 = self.channel_pjt(image2)
        
        else:
            if self.inside_fusion is None:
                fmap1, fmap2 = self.fnet([image1, image2])        
            elif self.inside_fusion == 'parallel':
                fmap1 = self.fnet(image1, mask1)
                fmap2 = self.fnet(image2, mask2)
        
        fmap1 = fmap1.float()
        self.fmap = fmap1
        fmap2 = fmap2.float()
        
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        if self.inside_fusion == 'parallel' and self.fuse_cnet:
            cnet = self.cnet(image1, mask1)
        else:
            cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
