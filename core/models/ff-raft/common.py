import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(seed, cudnn_enabled, allow_tf32, num_threads):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if cudnn_enabled:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    
    if allow_tf32:
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
    
    # torch.set_num_threads(num_threads)
    

def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        opt = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))
    opt.GLOBAL = argparse.Namespace(**opt.GLOBAL)
    opt.TRAIN = argparse.Namespace(**opt.TRAIN)
    opt.MODEL = argparse.Namespace(**opt.MODEL)
    opt.CRITERION = argparse.Namespace(**opt.CRITERION)
    opt.OPTIMIZER = argparse.Namespace(**opt.OPTIMIZER)
    opt.SCHEDULER = argparse.Namespace(**opt.SCHEDULER)

    return opt


def parallel_model(model, device, rank, local_rank):
    # DDP mode
    ddp_mode = device.type != 'cpu' and rank != -1
    if ddp_mode:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return model
