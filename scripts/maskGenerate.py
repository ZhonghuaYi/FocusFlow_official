import numpy as np
import torch
import torchvision
import os
from glob import glob
from pathlib import Path
import cv2 as cv
import sys
from tqdm import tqdm

class GoodFeatureCreator:
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Create mask using goodFeaturesToTrack.
        Args:
            img: image to be processed.

        Returns:
            mask: mask created by goodFeaturesToTrack.
        """
        mask = np.zeros(img.shape, dtype=np.uint8)
        kp = cv.goodFeaturesToTrack(img, 500, 0.01, 10)
        for i in kp:
            x = int(i[0][0])
            y = int(i[0][1])
            mask[y, x] = 255
        return mask


class SIFTFeatureCreator:
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Create mask using SIFT.
        Args:
            img: image to be processed.

        Returns:
            mask: mask created by SIFT.
        """
        mask = np.zeros(img.shape, dtype=np.uint8)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        for i in kp:
            x = int(i.pt[0])
            y = int(i.pt[1])
            mask[y, x] = 255
        return mask


class ORBFeatureCreator:
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Create mask using ORB.
        Args:
            img: image to be processed.

        Returns:
            mask: mask created by ORB.
        """
        mask = np.zeros(img.shape, dtype=np.uint8)
        orb = cv.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        for i in kp:
            x = int(i.pt[0])
            y = int(i.pt[1])
            mask[y, x] = 255
        return mask



class SiLKFeatureCreator:
    def __init__(self):
        pass

    def __call__(self, img):
        pass


class FlyingChairsMaskGenerator:
    def __init__(self, root, save_path):
        """
        FlyingChairsMaskGenerator is used to generate masks for FlyingChairs dataset.
        Args:
            root: root path of FlyingChairs dataset.
            save_path: path to save masks.
            feature_type: type of feature to be used to generate masks.
        """
        self.root = root
        self.save_path = save_path
        self.image_paths = glob(os.path.join(self.root, '*.ppm'))
        self.feature_creator = None

    def set_feature_creator(self, feature_type):
        """
        Set feature creator.
        Args:
            feature_type: type of feature to be used to generate masks.
        """
        if feature_type == 'goodfeature':
            self.feature_creator = GoodFeatureCreator()
        elif feature_type == 'sift':
            self.feature_creator = SIFTFeatureCreator()
        elif feature_type == 'orb':
            self.feature_creator = ORBFeatureCreator()
        elif feature_type == 'silk':
            self.feature_creator = SiLKFeatureCreator()
        else:
            raise ValueError('feature_type must be one of goodfeatures, sift, orb, silk.')

    def generate_masks(self, feature_type):
        """
        Generate masks for FlyingChairs dataset.
        """
        self.set_feature_creator(feature_type)
        masks_save_path = os.path.join(self.save_path, feature_type)
        if not os.path.exists(masks_save_path):
            os.makedirs(masks_save_path)

        pbar = tqdm(self.image_paths)
        for path in pbar:
            img_name = os.path.basename(path)[:-4]
            img = cv.imread(path, 0)

            mask = self.feature_creator(img)
            mask_path = os.path.join(self.save_path, img_name + '.png')
            cv.imwrite(mask_path, mask)

            pbar.set_description(f"Processing: {img_name}")


class FlyingThingsMaskGenerator:
    def __init__(self, root, save_path):
        """
        FlyingChairsMaskGenerator is used to generate masks for FlyingChairs dataset.
        Args:
            root: root path of FlyingChairs dataset.
            save_path: path to save masks.
            feature_type: type of feature to be used to generate masks.
        """
        self.root = root
        self.save_path = save_path
        self.feature_creator = None

    def set_feature_creator(self, feature_type):
        """
        Set feature creator.
        Args:
            feature_type: type of feature to be used to generate masks.
        """
        if feature_type == 'goodfeature':
            self.feature_creator = GoodFeatureCreator()
        elif feature_type == 'sift':
            self.feature_creator = SIFTFeatureCreator()
        elif feature_type == 'orb':
            self.feature_creator = ORBFeatureCreator()
        elif feature_type == 'silk':
            self.feature_creator = SiLKFeatureCreator()
        else:
            raise ValueError('feature_type must be one of goodfeatures, sift, orb, silk.')

    def generate_masks(self, feature_type):
        """
        Generate masks for FlyingChairs dataset.
        """
        self.set_feature_creator(feature_type)

        for pass_name in ("frames_cleanpass", "frames_finalpass"):
            for split in ("TRAIN", "TEST"):
                for class_name in ("A", "B", "C"):
                    for d in os.listdir(os.path.join(self.root, pass_name, split, class_name)):
                        for view in ("left", "right"):
                            image_root = os.path.join(self.root, pass_name, split, class_name, d, view)
                            mask_root = os.path.join(self.save_path, feature_type, pass_name, split, class_name, d, view)
                            if not os.path.exists(mask_root):
                                os.makedirs(mask_root)

                            print(f"Processing: {image_root}")
                            image_paths = glob(os.path.join(image_root, '*.png'))
                            pbar = tqdm(image_paths)
                            for path in pbar:
                                img_name = os.path.basename(path)[:-4]
                                img = cv.imread(path, 0)

                                mask = self.feature_creator(img)
                                mask_path = os.path.join(mask_root, img_name + '.png')
                                cv.imwrite(mask_path, mask)

                                pbar.set_description(f"Processing: {img_name}")


class SintelMaskGenerator:
    def __init__(self, root, save_path):
        """
        FlyingChairsMaskGenerator is used to generate masks for FlyingChairs dataset.
        Args:
            root: root path of FlyingChairs dataset.
            save_path: path to save masks.
            feature_type: type of feature to be used to generate masks.
        """
        self.root = root
        self.save_path = save_path
        self.feature_creator = None

    def set_feature_creator(self, feature_type):
        """
        Set feature creator.
        Args:
            feature_type: type of feature to be used to generate masks.
        """
        if feature_type == 'goodfeature':
            self.feature_creator = GoodFeatureCreator()
        elif feature_type == 'sift':
            self.feature_creator = SIFTFeatureCreator()
        elif feature_type == 'orb':
            self.feature_creator = ORBFeatureCreator()
        elif feature_type == 'silk':
            self.feature_creator = SiLKFeatureCreator()
        else:
            raise ValueError('feature_type must be one of goodfeatures, sift, orb, silk.')

    def generate_masks(self, feature_type):
        """
        Generate masks for FlyingChairs dataset.
        """
        self.set_feature_creator(feature_type)

        for split in ("training", "testing", "val"):
            for pass_name in ("clean", "final"):
                for d in os.listdir(os.path.join(self.root, split, pass_name)):
                    image_root = os.path.join(self.root, split, pass_name, d)
                    mask_root = os.path.join(self.save_path, feature_type, split, pass_name, d)
                    if not os.path.exists(mask_root):
                        os.makedirs(mask_root)

                        print(f"Processing: {image_root}")
                        image_paths = glob(os.path.join(image_root, '*.png'))
                        pbar = tqdm(image_paths)
                        for path in pbar:
                            img_name = os.path.basename(path)[:-4]
                            img = cv.imread(path, 0)

                            mask = self.feature_creator(img)
                            mask_path = os.path.join(mask_root, img_name + '.png')
                            cv.imwrite(mask_path, mask)

                            pbar.set_description(f"Processing: {img_name}")


class KITTIMaskGenerator:
    def __init__(self, root, save_path):
        """
        FlyingChairsMaskGenerator is used to generate masks for FlyingChairs dataset.
        Args:
            root: root path of FlyingChairs dataset.
            save_path: path to save masks.
            feature_type: type of feature to be used to generate masks.
        """
        self.root = root
        self.save_path = save_path
        self.feature_creator = None

    def set_feature_creator(self, feature_type):
        """
        Set feature creator.
        Args:
            feature_type: type of feature to be used to generate masks.
        """
        if feature_type == 'goodfeature':
            self.feature_creator = GoodFeatureCreator()
        elif feature_type == 'sift':
            self.feature_creator = SIFTFeatureCreator()
        elif feature_type == 'orb':
            self.feature_creator = ORBFeatureCreator()
        elif feature_type == 'silk':
            self.feature_creator = SiLKFeatureCreator()
        else:
            raise ValueError('feature_type must be one of goodfeatures, sift, orb, silk.')

    def generate_masks(self, feature_type):
        """
        Generate masks for FlyingChairs dataset.
        """
        self.set_feature_creator(feature_type)

        for split in ("training", "testing", "val"):
            image_root = os.path.join(self.root, split, 'image_2')
            mask_root = os.path.join(self.save_path, feature_type, split)
            if not os.path.exists(mask_root):
                os.makedirs(mask_root)

                print(f"Processing: {image_root}")
                image_paths = glob(os.path.join(image_root, '*.png'))
                pbar = tqdm(image_paths)
                for path in pbar:
                    img_name = os.path.basename(path)[:-4]
                    img = cv.imread(path, 0)

                    mask = self.feature_creator(img)
                    mask_path = os.path.join(mask_root, img_name + '.png')
                    cv.imwrite(mask_path, mask)

                    pbar.set_description(f"Processing: {img_name}")


if __name__ == '__main__':
    kitti_root = "datasets/KITTI-custom"
    kitti_masks_root = "../mask/KITTI"
    kitti_generator = KITTIMaskGenerator(kitti_root, kitti_masks_root)
    kitti_generator.generate_masks('goodfeature')
