import torch
import torch.nn as nn
from einops import rearrange

    
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
    

class CA(nn.Module):
    def __init__(self, in_channels, reduction=16, bias=True):
        super(CA, self).__init__()
        self.conv_q = nn.Conv2d(2*in_channels, in_channels, 3, padding=1, bias=bias)
        self.conv_v = self._projection_layer(in_channels, 3, bias)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.c_map = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        
    def _projection_layer(self, in_channels, kernel_size, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, bias=bias),
        )

    def forward(self, q, v):
        q1 = self.conv_q(torch.cat([q, v], dim=1))
        v = self.conv_v(v)
        q_avg = self.avgpool(q1)
        q_max = self.maxpool(q1)
        c_map = self.c_map(q_avg) + self.c_map(q_max)
        v = c_map * v
        out = v + q
        return out
    
    
class SA(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(SA, self).__init__()
        self.conv_q = nn.Conv2d(2*in_channels, in_channels, 3, padding=1, bias=bias)
        self.conv_v = self._projection_layer(in_channels, 3, bias)
        self.s_map = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=bias),
            nn.Sigmoid()
        )
        
    def _projection_layer(self, in_channels, kernel_size, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, bias=bias),
        )

    def forward(self, q, v):
        q1 = self.conv_q(torch.cat([q, v], dim=1))
        v = self.conv_v(v)
        q_mean = torch.mean(q1, dim=1, keepdim=True)
        q_max, _ = torch.max(q1, dim=1, keepdim=True)
        q1 = torch.cat([q_mean, q_max], dim=1)
        s_map = self.s_map(q1)
        v = s_map * v
        out = v + q
        return out
    
    
class Concat(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2*in_channels, in_channels, 1)
    
    def forward(self, q, v):
        v = torch.cat([q, v], dim=1)
        out = self.conv(v)
        return out
    

class Conv1x1(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, q, v):
        v = self.conv(v)
        out = q + v
        return out
    
    
class FusionUnit(nn.Module):
    def __init__(self, in_channels, fusion_type, bi_direction=True) -> None:
        super().__init__()
        self.bi_direction = bi_direction
        self.fusion_type = fusion_type
        self.res_link = True
        
        if fusion_type == '1x1conv':
            self.mask2img = Conv1x1(in_channels)
            if bi_direction:
                self.img2mask = Conv1x1(in_channels)
            else:
                self.img2mask = None
                
        elif fusion_type == '1x1conv-unidirection':
            self.mask2img = Conv1x1(in_channels)
            self.img2mask = None
                
        elif fusion_type == 'concat':
            self.mask2img = Concat(in_channels)
            if bi_direction:
                self.img2mask = Concat(in_channels)
            else:
                self.img2mask = None
                
        elif fusion_type == 'SA':
            self.mask2img = SA(in_channels)
            if bi_direction:
                self.img2mask = SA(in_channels)
            else:
                self.img2mask = None
                
        elif fusion_type == 'CA':
            self.mask2img = CA(in_channels)
            if bi_direction:
                self.img2mask = CA(in_channels)
            else:
                self.img2mask = None
                
        else:
            raise ValueError(f"Fusion type {fusion_type} not supported.")
            
            
    
    def forward(self, mask, img):
        img_out = self.mask2img(img, mask)
        
        if self.img2mask is not None:
            mask_out = self.img2mask(mask, img)
        else:
            mask_out = mask
            
        return mask_out, img_out
            
        


