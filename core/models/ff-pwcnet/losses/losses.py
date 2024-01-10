import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def get_kernel(kernel_size, sigma):
    sigma_3 = 3 * sigma
    X = np.linspace(-sigma_3, sigma_3, kernel_size)
    Y = np.linspace(-sigma_3, sigma_3, kernel_size)
    x, y = np.meshgrid(X, Y)
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
    Z = gauss.sum()
    kernel = (1/Z)*gauss
    return torch.FloatTensor(kernel).view(1, 1, kernel_size, kernel_size)


class EPELoss(nn.Module):
    def __init__(self, cfg):
        super(EPELoss, self).__init__()
        self.cfg = cfg
        self.loss_mode = cfg.TRAIN.LOSS_MODE
        self.weights = cfg.TRAIN.LOSS_WEIGHTS
        self.loss_q = cfg.TRAIN.LOSS_Q
        self.loss_epsilon = cfg.TRAIN.LOSS_EPSILON

    def EPE(self, input_flow, target_flow, sparse=False, mean=True):
        if self.loss_mode == 'pretrain':
            EPE_map = torch.norm(target_flow - input_flow, 2, 1)
        else:
            EPE_map = (torch.norm(target_flow - input_flow, 1, 1) + self.loss_epsilon) ** self.loss_q
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

            EPE_map = EPE_map[~mask]
        if mean:
            return EPE_map.mean()
        else:
            return EPE_map.sum() / batch_size

    def sparse_max_pool(self, input, size):
        '''Downsample the input by considering 0 values as invalid.

        Unfortunately, no generic interpolation mode can resize a sparse map correctly,
        the strategy here is to use max pooling for positive values and "min pooling"
        for negative values, the two results are then summed.
        This technique allows sparsity to be minized, contrary to nearest interpolation,
        which could potentially lose information for isolated data points.'''

        positive = (input > 0).float()
        negative = (input < 0).float()
        output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
        return output

    def multiscaleEPE(self, network_output, target_flow, sparse=False):
        def one_scale(output, target, sparse):

            b, _, h, w = output.size()

            if sparse:
                target_scaled = self.sparse_max_pool(target, (h, w))
            else:
                target_scaled = F.interpolate(target, (h, w), mode='area')
            return self.EPE(output, target_scaled, sparse, mean=False)

        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        assert (len(self.weights) == len(network_output))

        loss = 0
        for output, weight in zip(network_output, self.weights):
            loss += weight * one_scale(output, target_flow, sparse)
        return loss

    def realEPE(self, output, target, sparse=False):
        b, _, h, w = target.size()
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        return self.EPE(upsampled_output, target, sparse, mean=True)

    def forward(self, output, target, sparse=False):
        loss = self.multiscaleEPE(output, target, sparse)
        result = {'epe': self.realEPE(output[0], target, sparse), 'loss': loss.detach()}
        return loss, result


class CPCL(EPELoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kernel_size = cfg.TRAIN.LOSS_KERNEL_SIZE
        self.sigma = cfg.TRAIN.LOSS_SIGMA
        
    def EPE(self, input_flow, target_flow, mask, sparse=False, mean=True):
        _, _, h, w = input_flow.size()
        if self.loss_mode == 'pretrain':
            EPE_map = torch.norm(target_flow - input_flow, 2, 1)
        else:
            EPE_map = (torch.norm(target_flow - input_flow, 1, 1) + self.loss_epsilon) ** self.loss_q
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

            EPE_map = EPE_map[~mask]
            
        mask = (mask > 0).float()
        kernel = get_kernel(self.kernel_size, self.sigma).to(mask.device)
        pad = self.kernel_size // 2
        mask = F.pad(mask, [pad, pad, pad, pad])
        mask = F.conv2d(mask, kernel)
        
        EPE_map = EPE_map * mask
        
        if mean:
            return EPE_map.sum() / mask.sum()
        else:
            return EPE_map.sum() / mask.sum() * (h * w)
    
    def realEPE(self, output, target, sparse=False):
        b, _, h, w = target.size()
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        input_flow = upsampled_output
        target_flow = target
        
        if self.loss_mode == 'pretrain':
            EPE_map = torch.norm(target_flow - input_flow, 2, 1)
        else:
            EPE_map = (torch.norm(target_flow - input_flow, 1, 1) + self.loss_epsilon) ** self.loss_q
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

            EPE_map = EPE_map[~mask]
            
        return EPE_map.mean()
    
    def multiscaleEPE(self, network_output, target_flow, mask, sparse=False):
        def one_scale(output, target, mask, sparse):

            b, _, h, w = output.size()

            if sparse:
                target_scaled = self.sparse_max_pool(target, (h, w))
            else:
                target_scaled = F.interpolate(target, (h, w), mode='area')
                mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=False)
            return self.EPE(output, target_scaled, mask, sparse, mean=False)

        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        assert (len(self.weights) == len(network_output))

        loss = 0
        for output, weight in zip(network_output, self.weights):
            loss += weight * one_scale(output, target_flow, mask, sparse)
        return loss
        
    def forward(self, output, target, mask, sparse=False):
        loss = self.multiscaleEPE(output, target, mask, sparse)
        result = {'epe': self.realEPE(output[0], target, sparse), 'loss': loss.detach()}
        return loss, result
    

class MixLoss(EPELoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kernel_size = cfg.TRAIN.LOSS_KERNEL_SIZE
        self.sigma = cfg.TRAIN.LOSS_SIGMA
        self.lamda = cfg.TRAIN.LOSS_LAMDA
        
    def EPE(self, input_flow, target_flow, mask, sparse=False, mean=True):
        _, _, h, w = input_flow.size()
        if self.loss_mode == 'pretrain':
            EPE_map = torch.norm(target_flow - input_flow, 2, 1)
        else:
            EPE_map = (torch.norm(target_flow - input_flow, 1, 1) + self.loss_epsilon) ** self.loss_q
        
        EPE_map = EPE_map.unsqueeze(1)
        batch_size = EPE_map.size(0)
        
        mask = (mask > 0).float()
        if mask.sum() == 0:
            mask_EPE_map = 0
        else:
            kernel = get_kernel(self.kernel_size, self.sigma).to(mask.device)
            pad = self.kernel_size // 2
            mask = F.pad(mask, [pad, pad, pad, pad])
            mask = F.conv2d(mask, kernel)
            
            mask_EPE_map = EPE_map * mask
        
            if sparse:
                # invalid flow is defined with both flow coordinates to be exactly 0
                valid = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
                valid = valid.unsqueeze(1)
                mask_EPE_map = mask_EPE_map[~valid]
        
        if mean:
            epe = EPE_map.mean()
            if mask.sum() == 0:
                mask_epe = 0
            else:
                mask_epe = mask_EPE_map.sum() / mask.sum()
            return epe + self.lamda * mask_epe
        else:
            epe = EPE_map.sum()
            if mask.sum() == 0:
                mask_epe = 0
            else:
                mask_epe = mask_EPE_map.sum() / mask.sum() * (h * w)
            return epe + self.lamda * mask_epe
    
    def realEPE(self, output, target, sparse=False):
        b, _, h, w = target.size()
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        input_flow = upsampled_output
        target_flow = target
        
        if self.loss_mode == 'pretrain':
            EPE_map = torch.norm(target_flow - input_flow, 2, 1)
        else:
            EPE_map = (torch.norm(target_flow - input_flow, 1, 1) + self.loss_epsilon) ** self.loss_q
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

            EPE_map = EPE_map[~mask]
            
        return EPE_map.mean()
    
    def multiscaleEPE(self, network_output, target_flow, mask, sparse=False):
        def one_scale(output, target, mask, sparse):

            b, _, h, w = output.size()

            if sparse:
                target_scaled = self.sparse_max_pool(target, (h, w))
                mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=False)
            else:
                target_scaled = F.interpolate(target, (h, w), mode='area')
                mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=False)
            return self.EPE(output, target_scaled, mask, sparse, mean=False)

        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        assert (len(self.weights) == len(network_output))

        loss = 0
        for output, weight in zip(network_output, self.weights):
            loss += weight * one_scale(output, target_flow, mask, sparse)
        return loss
        
    def forward(self, output, target, mask, sparse=False):
        loss = self.multiscaleEPE(output, target, mask, sparse)
        result = {'epe': self.realEPE(output[0], target, sparse), 'loss': loss.detach()}
        return loss, result


if __name__ == '__main__':
    pass
