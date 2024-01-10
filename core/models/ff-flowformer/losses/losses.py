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
    def __init__(self, gamma=0.8, max_flow=400) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        
    def forward(self, flow_preds, flow_gt, valid, *args):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)    
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
        'epe': epe.mean().item(),
        'loss': flow_loss.detach().item(),
        }

        return flow_loss, metrics


class CPCL(nn.Module):
    
    def __init__(self, gamma=0.8, max_flow=400, kernel_size=5, sigma=1.7) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def forward(self, flow_preds, flow_gt, valid, mask):
        """ Loss function defined over sequence of flow predictions """
  
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        mask = (mask > 0).float()
        kernel = get_kernel(self.kernel_size, self.sigma).to(mask.device)
        pad = self.kernel_size // 2
        mask = F.pad(mask, [pad, pad, pad, pad])
        mask = F.conv2d(mask, kernel)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * mask * i_loss).sum() / mask.sum()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            'loss': flow_loss.detach().item(),
        }

        return flow_loss, metrics


class MixLoss(nn.Module):
    def __init__(self, gamma=0.8, max_flow=400, kernel_size=5, sigma=1.7, lamda=0.8) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lamda = lamda
        
    def forward(self, flow_preds, flow_gt, valid, mask):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        mask = (mask > 0).float()
        kernel = get_kernel(self.kernel_size, self.sigma).to(mask.device)
        pad = self.kernel_size // 2
        mask = F.pad(mask, [pad, pad, pad, pad])
        mask = F.conv2d(mask, kernel)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += self.lamda * i_weight * (valid[:, None] * mask * i_loss).sum() / mask.sum()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            'loss': flow_loss.detach().item(),
        }

        return flow_loss, metrics


if __name__ == '__main__':
    pass
