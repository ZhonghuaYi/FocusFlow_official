import torch
import torch.nn as nn
import timm
import numpy as np

class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
    
    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == layer-1:
                break
        
        return x

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num +=  np.prod(param.size())

            for param in drop.parameters():
                num +=  np.prod(param.size())

            for param in blocks.parameters():
                num +=  np.prod(param.size())

            for param in pos_blk.parameters():
                num +=  np.prod(param.size())

            if i == layer-1:
                break

        for param in self.svt.head.parameters():
            num +=  np.prod(param.size())
        
        return num

class twins_svt_large_context(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large_context', pretrained=pretrained)
    
    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == layer-1:
                break
        
        return x


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
                
        else:
            raise ValueError(f"Fusion type {fusion_type} not supported.")
    
    def forward(self, mask, img):
        img_out = self.mask2img(img, mask)
        
        if self.img2mask is not None:
            mask_out = self.img2mask(mask, img)
        else:
            mask_out = mask
            
        return mask_out, img_out


class twins_svt_large_CCE(twins_svt_large):
    def __init__(self):
        super().__init__(pretrained=True)
        self.mask_svt = timm.create_model('twins_svt_large', pretrained=False)

        del self.mask_svt.head
        del self.mask_svt.patch_embeds[2]
        del self.mask_svt.patch_embeds[2]
        del self.mask_svt.blocks[2]
        del self.mask_svt.blocks[2]
        del self.mask_svt.pos_block[2]
        del self.mask_svt.pos_block[2]
        
        self.fusion_blks = nn.ModuleList([
            FusionUnit(128, '1x1conv', True),
            FusionUnit(256, '1x1conv', False),
        ])
    
    def forward(self, x, mask, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, mask_embed, drop, mask_drop, blocks, mask_blocks, pos_blk, mask_pos_blk, fusion_blk) in enumerate(
            zip(self.svt.patch_embeds, self.mask_svt.patch_embeds, self.svt.pos_drops,  self.mask_svt.pos_drops, self.svt.blocks, self.mask_svt.blocks, self.svt.pos_block, self.mask_svt.pos_block, self.fusion_blks)):

            x, size = embed(x)
            mask, mask_size = mask_embed(mask)
            
            x = drop(x)
            mask = mask_drop(mask)
            for j, (blk, mask_blk) in enumerate(zip(blocks, mask_blocks)):
                x = blk(x, size)
                mask = mask_blk(mask, mask_size)
                if j==0:
                    x = pos_blk(x, size)
                    mask = mask_pos_blk(mask, mask_size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
                mask = mask.reshape(B, *mask_size, -1).permute(0, 3, 1, 2).contiguous()
                
            mask, x = fusion_blk(mask, x)
            
            if i == layer-1:
                break
        
        return x


if __name__ == "__main__":
    m = twins_svt_large()
    input = torch.randn(2, 3, 400, 800)
    mask = torch.randn(2, 3, 400, 800)
    # out, out1 = m(input, mask)
    print(m)
