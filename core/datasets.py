# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import sys

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, distributed
import torch.nn.functional as F
import cv2 as cv

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse

        # image augmentation
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []

    def __getitem__(self, index):

        # load images
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # load masks
        mask1 = frame_utils.read_gen(self.mask_list[index][0])
        mask2 = frame_utils.read_gen(self.mask_list[index][1])
        mask1 = np.array(mask1).astype(np.uint8)
        mask2 = np.array(mask2).astype(np.uint8)
        if mask1.ndim == 2:
            mask1 = mask1[..., None]
        elif mask1.ndim == 3:
            mask1 = mask1[..., :1]
            
        if mask2.ndim == 2:
            mask2 = mask2[..., None]
        elif mask2.ndim == 3:
            mask2 = mask2[..., :1]

        # not use augmentation for testing
        if self.is_test:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, mask1, mask2, self.extra_info[index]

        # set seed for random
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        # load flow. if sparse, load valid
        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        flow = np.array(flow).astype(np.float32)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid, mask1, mask2 = self.augmentor(img1, img2, flow, valid, mask1, mask2)
            else:
                img1, img2, flow, mask1, mask2 = self.augmentor(img1, img2, flow, mask1, mask2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        mask1 = torch.from_numpy(mask1).permute(2, 0, 1).float()
        mask2 = torch.from_numpy(mask2).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, mask1, mask2, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.mask_list = v * self.mask_list
        return self

    def __len__(self):
        return len(self.image_list)

class MpiSintel(FlowDataset):
    def __init__(self, root, mask_root, aug_params=None, split='training', dstype='clean', mask_type='orb'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        mask_root = osp.join(mask_root, mask_type, split, dstype)

        if split == 'testing':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            mask_list = sorted(glob(osp.join(mask_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.mask_list += [[mask_list[i], mask_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, root, mask_root, aug_params=None, split='training', mask_type='orb'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, 'data/*.ppm')))
        flows = sorted(glob(osp.join(root, 'data/*.flo')))
        masks = sorted(glob(osp.join(mask_root, mask_type, '*.png')))
        assert (len(images) == len(masks))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt(osp.join(root, 'FlyingChairs_train_val.txt'), dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]
                self.mask_list += [[masks[2 * i], masks[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, root, mask_root, aug_params=None, dstype='frames_cleanpass', mask_type='orb'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                mask_dirs = sorted(glob(osp.join(mask_root, mask_type, dstype, 'TRAIN/*/*')))
                mask_dirs = sorted([osp.join(f, cam) for f in mask_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir, mdir in zip(image_dirs, flow_dirs, mask_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    masks = sorted(glob(osp.join(mdir, '*.png')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.mask_list += [[masks[i], masks[i+1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.mask_list += [[masks[i+1], masks[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, root, mask_root, aug_params=None, split='training', mask_type='orb'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        image_root = osp.join(root, split)
        mask_root = osp.join(mask_root, mask_type, split)
        images1 = sorted(glob(osp.join(image_root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(image_root, 'image_2/*_11.png')))
        masks1 = sorted(glob(osp.join(mask_root, '*_10.png')))
        masks2 = sorted(glob(osp.join(mask_root, '*_11.png')))

        for img1, img2, mask1, mask2 in zip(images1, images2, masks1, masks2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
            self.mask_list += [[mask1, mask2]]

        if split == 'training' or 'val':
            self.flow_list = sorted(glob(osp.join(image_root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, root='datasets/HD1k', aug_params=None):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


class OminiFlow(FlowDataset):
    def __init__(self, root, aug_params=None):
        super(OminiFlow, self).__init__(aug_params)
        for scene in ("CartoonTree", "Forest", "lowPolyModels"):
            for split in ("0", "1", ):
                image_root = osp.join(root, scene)
                if split == "0":
                    image_root = osp.join(image_root, f"{scene}")
                else:
                    image_root = osp.join(image_root, f"{scene}_{split}")
                images = sorted(glob(osp.join(image_root, "images/*.png")))
                flows = sorted(glob(osp.join(image_root, "ground_truth/*.flo")))

                for i in range(len(images) - 1):
                    self.image_list += [[images[i], images[i + 1]]]
                    self.flow_list += [flows[i]]
                    self.mask_list += [[images[i], images[i + 1]]]


def fetch_dataloader(data_root, mask_root, cfg, rank=-1, world_size=1, TRAIN_DS=None):
    """ Create the data loader for the corresponding trainign set """

    if cfg.TRAIN.STAGE == 'chairs':
        aug_params = {'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(data_root["chairs"], mask_root["chairs"], aug_params=aug_params,
                                     split='training', mask_type=cfg.TRAIN.MASK_TYPE)

    elif cfg.TRAIN.STAGE == 'things':
        aug_params = {'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(data_root["things"], mask_root["things"], dstype='frames_cleanpass', aug_params=aug_params, mask_type=cfg.TRAIN.MASK_TYPE)
        final_dataset = FlyingThings3D(data_root["things"], mask_root["things"], dstype='frames_finalpass', aug_params=aug_params, mask_type=cfg.TRAIN.MASK_TYPE)
        train_dataset = clean_dataset + final_dataset

    elif cfg.TRAIN.STAGE == 'sintel':
        aug_params = {'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(data_root["sintel"], mask_root["sintel"], dstype='clean', aug_params=aug_params, mask_type=cfg.TRAIN.MASK_TYPE)
        sintel_final = MpiSintel(data_root["sintel"], mask_root["sintel"], dstype='final', aug_params=aug_params, mask_type=cfg.TRAIN.MASK_TYPE)

        if TRAIN_DS is not None:
            things = FlyingThings3D(data_root["things"], mask_root["things"], dstype='frames_cleanpass',
                                    aug_params=aug_params, mask_type=cfg.TRAIN.MASK_TYPE)

            if TRAIN_DS == 'C+T+S':
                train_dataset = 100 * sintel_clean + 100 * sintel_final + things

            elif TRAIN_DS == 'C+T+S+K':
                kitti_aug_params = {'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}
                kitti = KITTI(data_root["kitti"], mask_root["kitti"], split='training',
                              aug_params=kitti_aug_params, mask_type=cfg.TRAIN.MASK_TYPE)
                train_dataset = things + 100 * sintel_clean + 100 * sintel_final + 200 * kitti
        
        else:
            train_dataset = sintel_clean + sintel_final

    elif cfg.TRAIN.STAGE == 'kitti':
        # aug_params = {'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        # things = FlyingThings3D(data_root["things"], mask_root["things"], dstype='frames_cleanpass',
                                    # aug_params={'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}, mask_type=cfg.TRAIN.MASK_TYPE)
        sintel_clean = MpiSintel(data_root["sintel"], mask_root["sintel"], dstype='clean', aug_params={'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}, mask_type=cfg.TRAIN.MASK_TYPE)
        sintel_final = MpiSintel(data_root["sintel"], mask_root["sintel"], dstype='final', aug_params={'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}, mask_type=cfg.TRAIN.MASK_TYPE)
        kitti = KITTI(data_root["kitti"], mask_root["kitti"], split='training',
                              aug_params={'crop_size': cfg.TRAIN.IMAGE_SIZE, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}, mask_type=cfg.TRAIN.MASK_TYPE)
        train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti
        # train_dataset = kitti
        
    
    train_sampler = (None if rank == -1 else distributed.DistributedSampler(train_dataset, shuffle=True))
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.TRAIN.BATCH_SIZE // world_size,
                              pin_memory=True, 
                              shuffle=True and train_sampler is None, 
                              sampler=train_sampler,
                              num_workers=cfg.GLOBAL.NUM_WORKERS, 
                              drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

