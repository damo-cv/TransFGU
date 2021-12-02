'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/facebookresearch/dino/blob/main/video_generation.py
'''
import torch


def binarize_mask(mask, th=0.6):
    '''
    keep the first 'th' percent of value
    '''
    val, idx = torch.sort(mask)
    val /= torch.sum(val, dim=0, keepdim=True)
    cumval = torch.cumsum(val, dim=0)
    th_attn = cumval > (1 - th)
    idx2 = torch.argsort(idx)
    mask_bin = th_attn[idx2]

    return mask_bin