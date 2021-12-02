'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import random
import cv2
import numpy as np

import torch
import torch.nn.functional as F


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, size=None, shortest_edge=480, patch_size=8, fix_ratio=False, consist_orient=False, img_only=True):
        self.shortest_edge = shortest_edge
        self.patch_size = patch_size
        self.fix_ratio = fix_ratio
        self.consist_orient = consist_orient
        self.img_only = img_only
        self.ratio = 854/480
        self.size = size

    def __call__(self, sample):

        img = sample['img']
        label_cat = sample['label_cat']
        original_size = np.array(img.shape[0:2])
        if self.size is not None:
            target_size = self.size
        else:
            if not self.fix_ratio:
                ratio = original_size.max() / original_size.min()
            else:
                ratio = self.ratio
            if not self.consist_orient:
                target_size = (self.shortest_edge, int(self.shortest_edge * ratio)) if original_size[0] < original_size[1] \
                    else (int(self.shortest_edge * ratio), self.shortest_edge)
            else:
                target_size = (self.shortest_edge, int(self.shortest_edge * ratio))
            delta_h = target_size[0] // self.patch_size
            delta_w = target_size[1] // self.patch_size

            target_size = (delta_h * self.patch_size, delta_w * self.patch_size)
        if not tuple(original_size) == target_size:
            img = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_NEAREST)
            if not self.img_only:
                label_cat = cv2.resize(label_cat, target_size[::-1], interpolation=cv2.INTER_NEAREST)


        sample['img'] = img
        sample['label_cat'] = label_cat
        sample['meta']['original_size'] = original_size

        return sample



class ResizeTensor(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, size=None, shortest_edge=480, patch_size=8, fix_ratio=False, consist_orient=False, img_only=True):
        self.shortest_edge = shortest_edge
        self.patch_size = patch_size
        self.fix_ratio = fix_ratio
        self.consist_orient = consist_orient
        self.img_only = img_only
        self.ratio = 854/480
        self.size = size

    def __call__(self, sample):

        img = sample['img']
        label_cat = sample['label_cat']
        original_size = torch.tensor(img.shape[1:])
        if self.size is not None:
            target_size = self.size
        else:
            if not self.fix_ratio:
                ratio = original_size.max() / original_size.min()
            else:
                ratio = self.ratio
            target_size = (self.shortest_edge, int(self.shortest_edge * ratio)) if original_size[0] < original_size[1] \
                else (int(self.shortest_edge * ratio), self.shortest_edge)
            delta_h = target_size[0] // self.patch_size
            delta_w = target_size[1] // self.patch_size

            target_size = (delta_h * self.patch_size, delta_w * self.patch_size)

        if not tuple(original_size) == target_size:
            img = F.interpolate(img[None].float(), target_size, mode='bilinear')[0]
            if not self.img_only:
                label_cat = F.interpolate(label_cat[None, None].float(), target_size)[0, 0].int()


        sample['img'] = img
        sample['label_cat'] = label_cat
        sample['meta']['original_size'] = original_size

        return sample



class RandomCrop(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scale_min=0.4, scale_max=1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):

        img = sample['img']
        label_cat = sample['label_cat']
        pseudo_label = sample['pseudo_label'] if 'pseudo_label' in sample.keys() else None
        size = np.array(img.shape[0:2])
        scale = random.randint(self.scale_min*10, self.scale_max*10) / 10.

        crop_window_size = int(size.min() * scale)

        available_y, available_x = size[0] - crop_window_size, size[1] - crop_window_size

        y1, x1 = random.randint(0, available_y), random.randint(0, available_x)

        img_crop = img[y1:y1+crop_window_size, x1:x1+crop_window_size]
        label_cat_crop = label_cat[y1:y1+crop_window_size, x1:x1+crop_window_size]
        if 'pseudo_label' in sample.keys():
            pseudo_label_ = cv2.resize(pseudo_label.transpose((1, 2, 0)), size[::-1], interpolation=cv2.INTER_LINEAR)
            pseudo_label_ = cv2.resize(pseudo_label_[y1:y1+crop_window_size, x1:x1+crop_window_size],
                                       np.array(pseudo_label.shape[1:])[::-1],
                                       interpolation=cv2.INTER_LINEAR)
            sample['pseudo_label'] = pseudo_label_.transpose((2, 0, 1))
        sample['img'] = img_crop
        sample['label_cat'] = label_cat_crop

        return sample



class NormInput(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), zero_mean=False):
        self.mean = mean
        self.std = std
        self.zero_mean = zero_mean

    def __call__(self, sample):

        # --- fetch ---
        img = sample['img']


        if self.mean is None:
            img = img.astype(np.float32) / 127.5 - 1  # [-1, 1]
        else:
            img = ((img / 255) - self.mean) / self.std
        if self.zero_mean:
            pass

        sample['img'] = img
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img = sample['img']
        label_cat = sample['label_cat']

        img = torch.from_numpy(img).permute((2, 0, 1))
        label_cat = torch.from_numpy(np.ascontiguousarray(label_cat))

        sample['img'] = img
        sample['label_cat'] = label_cat

        return sample


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, sample):

        img = sample['img']
        img = img.astype(np.float64)
        v = np.random.uniform(self.lower, self.upper)
        img *= v
        img = img.astype(np.uint8)
        img = np.clip(img, 0, 255)
        sample['img'] = img

        return sample


class RandomMirror(object):
    """
    Randomly horizontally flip the video volume
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        v = random.randint(0, 1)
        if v == 0:
            sample['meta']['flip'] = 0
            return sample

        img = sample['img']
        label_cat = sample['label_cat']
        pseudo_label = sample['pseudo_label'] if 'pseudo_label' in sample.keys() else None

        img = img[:, ::-1, :]
        label_cat = label_cat[:, ::-1]
        if 'pseudo_label' in sample.keys():
            sample['pseudo_label'] = pseudo_label[:, :, ::-1].copy()

        sample['img'] = img
        sample['label_cat'] = label_cat
        sample['meta']['flip'] = 1
        return sample


class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, sample):

        img = sample['img']

        img = img.astype(np.float64)
        v = np.random.uniform(-self.delta, self.delta)
        img += v
        img = img.astype(np.uint8)
        img = np.clip(img, 0, 255)
        sample['img'] = img

        return sample
