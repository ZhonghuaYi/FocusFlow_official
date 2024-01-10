import torch
import torch.nn as nn
from .extractor import BasicEncoder, ResidualBlock
from .attention import MaskImage_ChannelAttentionTransformerBlock
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
        # return nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, bias=bias),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, bias=bias)
        # )
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
            

class BasicParallelFusionLayer(BasicEncoder):
    def __init__(self, img_channel=3, mask_channel=3, output_dim=128, norm_fn='batch', dropout=0, cfg=None) -> None:
        super().__init__(img_channel, output_dim, norm_fn, dropout)
        self.fusion_type = cfg.MODEL.FUSION_TYPE
        print(f"Parallel fusion type: {self.fusion_type}")

        if self.norm_fn == 'group':
            self.mask_norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.mask_norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.mask_norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.mask_norm1 = nn.Sequential()
        
        self.mask_conv1 = nn.Conv2d(mask_channel, 64, kernel_size=7, stride=2, padding=3)
        self.mask_relu1 = nn.ReLU(inplace=True)
        self.fusion1 = FusionUnit(64, self.fusion_type, True)
        self.fusion2 = FusionUnit(64, self.fusion_type, True)
        self.fusion3 = FusionUnit(96, self.fusion_type, True)
        self.fusion4 = FusionUnit(128, self.fusion_type, True)
        self.fusion5 = FusionUnit(output_dim, self.fusion_type, False)

        self.mask_in_planes = 64
        self.mask_layer1 = self._make_mask_layer(64,  stride=1)
        
        self.mask_layer2 = self._make_mask_layer(96, stride=2)
        
        self.mask_layer3 = self._make_mask_layer(128, stride=2)

        # output convolution
        self.mask_conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
                    
    def _make_mask_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.mask_in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.mask_in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, x, mask):
        
        mask = self.mask_conv1(mask)
        mask = self.mask_norm1(mask)
        mask = self.mask_relu1(mask)
        
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        mask, x = self.fusion1(mask , x)
        
        mask = self.mask_layer1(mask)
        x = self.layer1(x)
        
        mask, x = self.fusion2(mask, x)
        
        mask = self.mask_layer2(mask)
        x = self.layer2(x)
        
        mask, x = self.fusion3(mask, x)
        
        mask = self.mask_layer3(mask)
        x = self.layer3(x)
        
        mask, x = self.fusion4(mask, x)
        
        mask = self.mask_conv2(mask)
        x = self.conv2(x)

        mask, x = self.fusion5(mask, x)
        
        if self.training and self.dropout is not None:
            x = self.dropout(x)
            
        return x
    
    def freeze_self(self, mode):
        if mode == 'parallel':
            for name, parameter in self.conv1.named_parameters():
                parameter.requires_grad = False
            
            for name, parameter in self.norm1.named_parameters():
                parameter.requires_grad = False
                
            for name, parameter in self.layer1.named_parameters():
                parameter.requires_grad = False
                
            for name, parameter in self.layer2.named_parameters():
                parameter.requires_grad = False
                
            for name, parameter in self.layer3.named_parameters():
                parameter.requires_grad = False
                
            for name, parameter in self.conv2.named_parameters():
                parameter.requires_grad = False
                
    def copy_to_branch(self):
        self.mask_conv1.load_state_dict(self.conv1.state_dict())
        self.mask_layer1.load_state_dict(self.layer1.state_dict())
        self.mask_layer2.load_state_dict(self.layer2.state_dict())
        self.mask_layer3.load_state_dict(self.layer3.state_dict())
        self.mask_conv2.load_state_dict(self.conv2.state_dict())
        


