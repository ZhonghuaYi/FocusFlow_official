import torch

def build_flowformer(cfg):
    if cfg.FUSION == 'parallel':
        from .LatentCostFormer.transformer import FF_FlowFormer
        return FF_FlowFormer(cfg)
    else:
        from .LatentCostFormer.transformer import FlowFormer
        return FlowFormer(cfg)