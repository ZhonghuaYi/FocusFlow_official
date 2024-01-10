import os
import time
import datetime
import logging
from shutil import copyfile, copytree

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import sys
import argparse

sys.path.append("../../../core")

from common_util import yaml_parser, setup, parallel_model

from FF_FlowFormer_Core.utils.misc import process_cfg
from FF_FlowFormer_Core.FlowFormer import build_flowformer
from FF_FlowFormer_Core.optimizer import fetch_optimizer

import datasets
import evaluate
from losses import build_losses

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler


class Logger:
    def __init__(self, log_dir, yaml):
        self.total_steps = 0
        self.running_loss = {}
        self.MAX_FLOW = 400
        self.SUM_FREQ = 100
        self.VAL_FREQ = 5000
        
        self.log_dir = log_dir
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        copyfile("./train.py", f"{self.log_dir}/train.py")
        copytree("./FF_FlowFormer_Core/", f"{self.log_dir}/core/", dirs_exist_ok=True)
        copyfile(yaml, f"{self.log_dir}/config.yaml")
        self.logger = self._init_logger()

    def _print_training_status(self):

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0
            
    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    
    def log_info(self, info):
        self.logger.info(info)

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            print(key, results[key])
            self.log_info(f"{key}-{results[key]}")
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_envs():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size

def set_cuda_devices(gpus):
    if isinstance(gpus, list):
        devices = ''
        for i in range(len(gpus)):
            if i > 0:
                devices += ", "
            devices += str(gpus[i])
            
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def train(args, cfg):
    set_cuda_devices(args.gpus)
    
    rank, local_rank, world_size = get_envs()
    if rank in [-1, 0]:
        print(f"DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    if local_rank != -1:  # DDP distriuted mode
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                init_method='env://', rank=local_rank, world_size=world_size)
    
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    LOG_DIR = os.path.join('runs', current_time)
    if cfg is not None:
        LOG_DIR = os.path.join('runs', current_time + '_' + cfg.GLOBAL.NAME)
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(f"{LOG_DIR}/checkpoints", exist_ok=True)
        if rank in [-1, 0]:
            print(f"Log directory: {LOG_DIR}")

    logger = Logger(LOG_DIR, args.yaml)
    
    if rank in [-1, 0]:
        print(f"current experiment: {cfg.GLOBAL.NAME}")
        logger.log_info(f"current experiment: {cfg.GLOBAL.NAME}")
        
    data_root = {}
    mask_root = {}

    data_root["chairs"] = "../../../data/FlyingChairs"
    mask_root["chairs"] = "../../../data/mask/FlyingChairs_release"

    data_root["things"] = "../../../data/FlyingThings3D"
    mask_root["things"] = "../../../data/mask/FlyingThings3D"

    data_root["sintel"] = "../../../data/Sintel-custom"
    mask_root["sintel"] = "../../../data/mask/Sintel-custom"

    data_root["kitti"] = "../../../data/KITTI-custom"
    mask_root["kitti"] = "../../../data/mask/KITTI-custom"

    train_loader = datasets.fetch_dataloader(data_root, mask_root, cfg, rank, world_size, "C+T+S")

    if rank in [-1, 0]:
        print("\rDataset loaded.") 

    if rank in [-1, 0]:
        print(f"Model config : {cfg.MODEL}")
        logger.log_info(f"Model config : {cfg.MODEL}")
    
    model = build_flowformer(cfg.MODEL)
    # model.load_state_dict(torch.load("models/ffraft-fusion-conv-chairs.pth"))
    if rank in [-1, 0]:
        print("\rModel initialized.")

    total_steps = 0

    if cfg.TRAIN.RESTORE_CHECKPOINT is not None:
        checkpoint = torch.load(cfg.TRAIN.RESTORE_CHECKPOINT)
        model.load_state_dict(checkpoint['model'], strict=True)
        total_steps = checkpoint['step']
        if rank in [-1, 0]:
            print(f"Load checkpoint from {cfg.TRAIN.RESTORE_CHECKPOINT}")
            logger.log_info(f"Load checkpoint from {cfg.TRAIN.RESTORE_CHECKPOINT}")

    if rank in [-1, 0]:
        print("Parameter Count: %d\n" % count_parameters(model))
        logger.log_info("Parameter Count: %d\n" % count_parameters(model))

    model.to(device)

    print(f"Trainer config: {cfg.TRAINER}")
    optimizer, scheduler = fetch_optimizer(model, cfg.TRAINER)
        
    if cfg.TRAIN.RESTORE_CHECKPOINT is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if rank in [-1, 0]:
        print(f"Total steps: {cfg.TRAIN.NUM_STEPS}")
        logger.log_info(f"Total steps: {cfg.TRAIN.NUM_STEPS}")

    if cfg.TRAIN.RESTORE_CHECKPOINT is not None:
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps - checkpoint['step'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    loss_function = build_losses(cfg.TRAIN.LOSS_TYPE, gamma=cfg.TRAIN.LOSS_GAMMA, max_flow=cfg.TRAIN.MAX_FLOW, kernel_size=cfg.TRAIN.LOSS_KERNEL_SIZE, sigma=cfg.TRAIN.LOSS_SIGMA, lamda=cfg.TRAIN.LOSS_LAMDA)
    
    if rank in [-1, 0]:
        print(f"Loss fuction: {type(loss_function)}--lamda: {cfg.TRAIN.LOSS_LAMDA}")
        logger.log_info(f"Loss fuction: {type(loss_function)}")
    
    model = parallel_model(model, device, rank, local_rank)

    scaler = GradScaler()
    if rank in [-1, 0]:
        print(f"Mixed precision: {cfg.GLOBAL.MIXED_PRECISION}")
        logger.log_info(f"\nMixed precision: {cfg.GLOBAL.MIXED_PRECISION}\n")

    VAL_FREQ = 5000

    time_start = time.time()

    should_keep_training = True
    current_training_step = 0
    while should_keep_training:
        for step, data_blob in enumerate(train_loader):
            if rank in [-1, 0]:
                print(f"\rWorking on {total_steps+1}/{cfg.TRAIN.NUM_STEPS}", end=" ")
            model.train()
            
            if rank != -1:
                train_loader.sampler.set_epoch(total_steps)
                
            optimizer.zero_grad()
            image1, image2, flow, mask1, mask2, valid = [x.cuda() for x in data_blob]

            if cfg.TRAIN.ADD_NOISE:
                stdv = np.random.uniform(0., 5.)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda(image1.device)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda(image2.device)).clamp(0.0, 255.0)

            with torch.cuda.amp.autocast(enabled=cfg.GLOBAL.MIXED_PRECISION):
                flow_predictions = model(image1, image2, mask1, mask2)
                loss, metrics = loss_function(flow_predictions, flow, valid, mask1)
                
            if rank != -1:
                loss *= world_size

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAINER.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            
            scaler.update()
            
            if rank in [-1, 0]:

                metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                logger.push(metrics)

                if total_steps % VAL_FREQ == VAL_FREQ - 1:
                    PATH = f'{LOG_DIR}/checkpoints/{total_steps + 1}_{cfg.GLOBAL.NAME}.pth.tar'
                    checkpoint = {
                        'step': total_steps + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    torch.save(checkpoint, PATH)
                    
                    results = {}
                    for val_dataset in cfg.CRITERION.VAL_DATASET:
                        logger.log_info(f"{total_steps}:---")
                        if rank in [-1, 0]:
                            if val_dataset == 'chairs':
                                results.update(evaluate.validate_chairs(model, cfg.TRAIN.MASK_TYPE))
                            elif val_dataset == 'sintel':
                                results.update(evaluate.validate_sintel(model, cfg.TRAIN.MASK_TYPE))
                            elif val_dataset == 'kitti':
                                results.update(evaluate.validate_kitti(model,
                                                                    cfg.TRAIN.MASK_TYPE))

                    logger.write_dict(results)

            total_steps += 1
            current_training_step += 1
            if current_training_step % 100 == 99:
                time_end = time.time()
                avg_use_time = (time_end - time_start) / 100
                if rank in [-1, 0]:
                    print(f"\tOne step used time: {round(avg_use_time, 5)}", end=' ')
                    logger.log_info(f"\nOne step used time: {avg_use_time}")
                last_time = avg_use_time * (cfg.TRAIN.NUM_STEPS - total_steps)
                if rank in [-1, 0]:
                    print(f"\tRest work will use: {str(datetime.timedelta(seconds=int(round(last_time))))}", end='')
                    logger.log_info(f"Rest work will use: {str(datetime.timedelta(seconds=int(round(last_time))))}\n")
                time_start = time.time()

            if total_steps >= cfg.TRAIN.NUM_STEPS:
                should_keep_training = False
                break
            
    for val_dataset in cfg.CRITERION.VAL_DATASET:
        logger.log_info(f"Final:---")
        print(f"Final:---")
        if rank in [-1, 0]:
            if val_dataset == 'chairs':
                result = evaluate.validate_chairs(model, cfg.TRAIN.MASK_TYPE)
                print(result)
                logger.log_info(result)
            elif val_dataset == 'sintel':
                result = evaluate.validate_sintel(model, cfg.TRAIN.MASK_TYPE)
                print(result)
                logger.log_info(result)
            elif val_dataset == 'kitti':
                result = evaluate.validate_kitti(model, cfg.TRAIN.MASK_TYPE)
                print(result)
                logger.log_info(result)

    if rank in [-1, 0]:
        logger.close()
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        PATH = f'{LOG_DIR}/{cfg.GLOBAL.NAME}_{current_time}.pth'
        torch.save(model.state_dict(), PATH)
    
    # destroy process
    if world_size > 1 and rank == 0:
        dist.destroy_process_group()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default='config/train.yaml', help="config file")
    parser.add_argument("--gpus", default=0, nargs='+', help="use which gpu")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()
    
    cfg = yaml_parser(args.yaml)

    setup(cfg.GLOBAL.SEED, cfg.GLOBAL.CUDNN_ENABLED, cfg.GLOBAL.ALLOW_TF32, cfg.GLOBAL.NUM_THREADS)

    train(args, cfg)
