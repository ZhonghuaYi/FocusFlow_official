from .losses import EPELoss, CPCL, MixLoss

def build_losses(loss_type, gamma=0.8, max_flow=400, kernel_size=5, sigma=1.7, lamda=0.8, **kwargs):
    if loss_type == 'EPELoss':
        return EPELoss(gamma, max_flow)
    elif loss_type == 'CPCL':
        return CPCL(gamma, max_flow, kernel_size, sigma)
    elif loss_type == 'MixLoss':
        return MixLoss(gamma, max_flow, kernel_size, sigma, lamda)
    else:
        raise ValueError(f'"loss_type":"{loss_type}" is not supported.')
    