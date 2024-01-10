import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import sys
import argparse

sys.path.append("../../../core")

import datasets
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def validate_chairs(model, mask_type):
    model.eval()
    print("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
    aepe_list = []
    mepe_list = []

    val_dataset = datasets.FlyingChairs("../../../data/FlyingChairs_release", "../../../data/mask/FlyingChairs_release",
                                        split="validation", mask_type=mask_type)
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=8, drop_last=False)
    for _, val_data in enumerate(val_loader):
        image1, image2, flow_gt, mask1, mask2, _ = [x.cuda() for x in val_data]

        _, flow_pr = model(image1, image2, mask1, mask2, raft_iters=12, test_mode=True)
        epe = (flow_pr - flow_gt) ** 2
        aepe = torch.sum(epe, dim=1).sqrt()
        aepe_list.append(aepe.view(-1).cpu().numpy())

        mask1 = (mask1 > 0.5).view(-1)
        mepe = aepe.view(-1)[mask1].mean()
        if not torch.isnan(mepe):
            mepe_list.append(mepe.cpu().numpy())

    aepe = np.mean(np.concatenate(aepe_list))
    mepe = np.mean(np.array(mepe_list))
    # print("Validation Chairs EPE: %f" % epe)
    return {'chairs': aepe, f'chairs-{mask_type}': mepe}


@torch.no_grad()
def validate_sintel(model, mask_type):
    model.eval()
    print("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
    results = {}
    for dstype in ["clean", "final"]:
        val_dataset = datasets.MpiSintel("../../../data/Sintel-custom", "../../../data/mask/Sintel-custom",
                                         dstype=dstype, mask_type=mask_type, split="val")
        val_loader = data.DataLoader(val_dataset, batch_size=4, pin_memory=False, shuffle=False, num_workers=4, drop_last=False)

        aepe_list = []
        mepe_list = []

        for _, val_data in enumerate(val_loader):
            image1, image2, flow_gt, mask1, mask2, _ = [x.cuda() for x in val_data]

            padder = InputPadder(image1.shape)
            image1, image2, mask1, mask2 = padder.pad(image1, image2, mask1, mask2)

            _, flow_pr = model(image1, image2, mask1, mask2, raft_iters=32, test_mode=True)
            flow = padder.unpad(flow_pr)
            mask1 = padder.unpad(mask1)

            epe = (flow - flow_gt) ** 2
            aepe = torch.sum(epe, dim=1).sqrt()
            aepe_list.append(aepe.view(-1).cpu().numpy())

            mask1 = (mask1 > 0.5).view(-1)
            mepe = aepe.view(-1)[mask1].mean()
            if not torch.isnan(mepe):
                mepe_list.append(mepe.cpu().numpy())

        aepe = np.mean(np.concatenate(aepe_list))
        mepe = np.mean(np.array(mepe_list))

        # print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[f'sintel-{dstype}'] = aepe
        results[f'sintel-{dstype}-{mask_type}'] = mepe

    return results


@torch.no_grad()
def validate_kitti(model, mask_type):
    model.eval()
    print("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))

    val_dataset = datasets.KITTI("../../../data/KITTI-custom", "../../../data/mask/KITTI-custom", split="val", mask_type=mask_type)
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=8, drop_last=False)

    out_list, aepe_list = [], []
    mepe_list = []
    for _, val_data in enumerate(val_loader):
        image1, image2, flow_gt, mask1, mask2, valid_gt = [x.cuda() for x in val_data]

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2, mask1, mask2 = padder.pad(image1, image2, mask1, mask2)

        _, flow_pr = model(image1, image2, mask1, mask2, raft_iters=32, test_mode=True)
        flow = padder.unpad(flow_pr)
        mask1 = padder.unpad(mask1)

        epe = (flow - flow_gt) ** 2
        aepe = torch.sum(epe, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

        epe = aepe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        mask1 = (mask1 > 0.5).view(-1)
        mepe = epe[mask1 * val].mean()
        if not torch.isnan(mepe):
            mepe_list.append(mepe.cpu().numpy())

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        aepe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    aepe_list = np.array(aepe_list)
    out_list = np.concatenate(out_list)

    aepe = np.mean(aepe_list)
    f1 = 100 * np.mean(out_list)
    mepe = np.mean(np.array(mepe_list))

    # print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': aepe, 'kitti-f1': f1, f'kitti-{mask_type}': mepe}

