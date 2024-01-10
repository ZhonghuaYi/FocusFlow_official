import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MaskImage_ChannelAttentionTransformerBlock


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample, relu_slope, use_emgc=False, num_heads=None):
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_channel, out_channel, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = MaskImage_ChannelAttentionTransformerBlock(out_channel,
                                                                                      num_heads=self.num_heads,
                                                                                      ffn_expansion_factor=4,
                                                                                      bias=False,
                                                                                      LayerNorm_type='WithBias')

    def forward(self, x, enc=None, dec=None, mask=None, mask_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
            out = out + out_enc + out_dec

        if mask_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, mask_filter)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.image_event_transformer(out_down, mask_filter)

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, mask_filter)
                return out


class UNetMaskConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample, relu_slope, use_emgc=False):
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_channel, out_channel, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if self.downsample:

            out_down = self.downsample(out)

            if not merge_before_downsample:

                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_channel, out_channel, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_channel, out_channel, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class FusionLayer(nn.Module):
    def __init__(self, img_channel=3, mask_channel=1, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2,
                 num_heads=[1, 2, 4]):
        super().__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(img_channel, wf, 3, 1, 1)
        # mask
        self.down_path_mask = nn.ModuleList()
        self.conv_mask0 = nn.Conv2d(mask_channel+img_channel, 8, 1)
        self.conv_mask1 = nn.Conv2d(8, wf, 3, 1, 1)

        prev_channels = wf

        for i in range(depth):
            downsample = True if (i+1) < depth else False

            self.down_path_1.append(UNetConvBlock(prev_channels,
                                                (2**i) * wf,
                                                downsample,
                                                relu_slope,
                                                num_heads=self.num_heads[i]))
            # mask encoder
            if i < depth:
                self.down_path_mask.append(UNetMaskConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope))

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i) * wf

        self.sam12 = SAM(prev_channels)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        image = x

        masks = []
        # mask encoder
        mask = torch.cat([mask, x], 1)
        mask = self.conv_mask0(mask)
        mask1 = self.conv_mask1(mask)
        for i, down in enumerate(self.down_path_mask):
            if i < self.depth-1:
                mask1, mask1_up = down(mask1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    masks.append(mask1_up)
                else:
                    masks.append(mask1)
            else:
                mask1 = down(mask1, self.fuse_before_downsample)
                masks.append(mask1)

        # fusion stage
        x1 = self.conv_01(image)
        encs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:

                x1, x1_up = down(x1, mask_filter=masks[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

            else:
                x1 = down(x1, mask_filter=masks[i], merge_before_downsample=self.fuse_before_downsample)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))

        sam_feature, _ = self.sam12(x1, image)

        return sam_feature


if __name__ == '__main__':
    pass

