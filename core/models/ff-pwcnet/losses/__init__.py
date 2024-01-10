from .losses import EPELoss, CPCL, MixLoss

def build_losses(cfg):
    loss_type = cfg.TRAIN.LOSS_TYPE
    if loss_type == 'EPELoss':
        return EPELoss(cfg)
    elif loss_type == 'CPCL':
        return CPCL(cfg)
    elif loss_type == 'MixLoss':
        return MixLoss(cfg)
    else:
        raise ValueError(f'"loss_type":"{loss_type}" is not supported.')
    