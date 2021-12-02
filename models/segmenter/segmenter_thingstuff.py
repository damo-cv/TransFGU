'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/rstrudel/segmenter/blob/master/segm/model/segmenter.py
'''
import random

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as T


import models.vision_transformer.vision_transformer as vits
from models.mask_transformer.decoder import MaskTransformer as Decoder
import models.utils.utils_transformer as utils_vits
from utils.binarize_mask import binarize_mask

from models.loss.CLSEmbLoss import cls_emb_loss
from models.loss.BootstrappingLoss import thingstuff_bootstrap_loss


from mmcv.ops.roi_align import roi_align

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class Segmenter(nn.Module):
    """ Vision Transformer """
    def __init__(self, cfg):
        super().__init__()
        self.n_things = cfg.decoder.n_things
        self.n_stuff = cfg.decoder.n_stuff
        self.n_cls = cfg.decoder.n_things + cfg.decoder.n_stuff
        self.n_cls_gt = cfg.decoder.n_things + cfg.decoder.n_stuff
        self.encoder = vits.__dict__[cfg.encoder.arch](patch_size=cfg.encoder.patch_size, num_classes=0)
        self.encoder_teacher = vits.__dict__[cfg.encoder.arch](patch_size=cfg.encoder.patch_size, num_classes=0)
        print(f"encoder {cfg.encoder.arch} {cfg.encoder.patch_size}x{cfg.encoder.patch_size} built.")
        utils_vits.load_pretrained_weights(self.encoder, cfg.encoder.pretrained_weight, cfg.encoder.arch,
                                           cfg.encoder.patch_size)
        self.decoder = Decoder(n_cls=self.n_cls, patch_size=cfg.encoder.patch_size)
        self.decoder_teacher = Decoder(n_cls=self.n_cls, patch_size=cfg.encoder.patch_size)

        self.fix_encoder = cfg.encoder.fix == 1

        self.encoder_teacher.load_state_dict(copy.deepcopy(self.encoder.state_dict()))
        self.decoder_teacher.load_state_dict(copy.deepcopy(self.decoder.state_dict()))
        self.encoder_teacher.eval()
        self.decoder_teacher.eval()

        self.img_aug = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
        )


    def get_grad_mask(self, x, cluster_centroids, th=0.1, mask_aggregate=False):

        N, _, H, W = x.shape
        h, w = H // self.decoder.patch_size, W // self.decoder.patch_size
        N_cls = self.n_cls
        embed_dim = self.encoder.embed_dim
        assert cluster_centroids.shape == (1, N_cls, embed_dim)
        cluster_centroids = cluster_centroids[0]
        self.encoder.eval()

        feats, attns = self.encoder.get_intermediate_layers_feat_attn(x, 1)
        attns[-1].retain_grad()
        cls_feat = feats[-1][:, 0]

        attn_vis = attns[-1].mean(1)[:, 0, 1:].view(N, h, w)
        attn_vis = attn_vis[:, None] / attn_vis.view(N, -1).max(-1)[0].view(N, 1, 1, 1)

        attn_grad_vis_list = []
        for cls_id in range(N_cls):
            label_cls_feat = cluster_centroids[cls_id][None]

            self.encoder.zero_grad()
            attns[-1].grad = None

            cls_feat_ = cls_feat / cls_feat.norm(dim=1, keepdim=True)
            label_cls_feat_ = label_cls_feat / label_cls_feat.norm(dim=1, keepdim=True)
            loss = (1 - cls_feat_ @ label_cls_feat_.transpose(1, 0)).mean()

            loss.backward(retain_graph=True)

            attn_grad_vis = attns[-1].grad.mean(1)[:, 0, 1:].view(N, 1, h, w)

            attn_grad_fg = -attn_grad_vis
            attn_grad_fg[attn_grad_fg < 0] = 0
            attn_grad_fg_bin = torch.stack([binarize_mask(attn_grad.flatten(), 0.8).view(h, w) for attn_grad in attn_grad_fg]).unsqueeze(1)
            attn_grad_fg[~attn_grad_fg_bin] = 0

            attn_grad_vis_ = attn_vis + attn_grad_fg
            attn_grad_vis_[attn_grad_vis > 0] = 0

            attn_grad_vis_list.append(attn_grad_vis_.detach())

        self.encoder.train()
        self.encoder.zero_grad()
        attns[-1].grad = None
        with torch.no_grad():
            attn_grad_vis_list = torch.cat(attn_grad_vis_list, 1)
            if not mask_aggregate:
                return attn_grad_vis_list.detach()
            else:
                raise NotImplementedError

    def data_augment(self, img, label, pseudo_label):

        device = img.device
        N, _, H, W = img.shape
        h, w = pseudo_label.shape[-2:]
        H_target, W_target = H // 2, W // 2
        h_pseudo_label, w_pseudo_label = H_target // self.decoder.patch_size, W_target // self.decoder.patch_size

        img_aug = self.img_aug(img / 255.)


        # ====== random resized crop ======
        scale_min, scale_max = 0.4, 1
        scales = torch.randint(int(scale_min*10), int(scale_max*10), (N, 1))[:, 0] / 10.
        crop_window_size_h, crop_window_size_w = (torch.tensor(H) * scales).int(), (torch.tensor(W) * scales).int()
        available_y, available_x = H - crop_window_size_h, W - crop_window_size_w

        y1, x1 = (available_y * torch.rand(N, 1)[:, 0]).int(), (available_x * torch.rand(N, 1)[:, 0]).int()

        img_crops = []
        label_crops = []
        for img_aug_, label_, y_, x_, crop_window_size_h_, crop_window_size_w_ in zip(img_aug, label, y1, x1, crop_window_size_h, crop_window_size_w):
            img_crops.append(F.interpolate(img_aug_[:, y_:y_+crop_window_size_h_, x_:x_+crop_window_size_w_][None], (H_target, W_target), mode='bilinear')[0])
            label_crops.append(F.interpolate(label_[y_:y_+crop_window_size_h_, x_:x_+crop_window_size_w_][None, None], (H_target, W_target))[0, 0])
        img_crops = torch.stack(img_crops)
        label_crops = torch.stack(label_crops)

        rois = torch.stack([x1, y1, x1+crop_window_size_w, y1+crop_window_size_h], 1) * (h / H)
        rois = torch.cat([torch.range(0, N-1)[:, None], rois], 1).to(device)
        pseudo_label_crop = roi_align(pseudo_label, rois, (h_pseudo_label, w_pseudo_label),
                            1.0, 0, 'avg', True).squeeze(1)

        # ====== random flip ======
        flag = torch.rand(N)
        for n in range(N):
            if flag[n] < 0.5:
                img_crops[n] = img_crops[n][:, :, range(img_crops.shape[-1]-1, -1, -1)]
                label_crops[n] = label_crops[n][:, range(label_crops.shape[-1]-1, -1, -1)]
                pseudo_label_crop[n] = pseudo_label_crop[n][:, :, range(pseudo_label_crop.shape[-1]-1, -1, -1)]

        return img_crops, label_crops, pseudo_label_crop


    def forward(self, x, label=None, pseudo_labels=None, return_loss=False, cluster_centroids=None, bootstrapping=False, augment=False, epoch=None):
        '''
        x:      N x 3 x H x W
        label:  N x H x W
        '''
        if augment:
            x, label, pseudo_labels = self.data_augment(x, label, pseudo_labels)
        N, _, H, W = x.shape
        h, w = H // self.decoder.patch_size, W // self.decoder.patch_size
        N_cls_fgbg = self.n_cls
        th = 0.1


        if not cluster_centroids is None:
            cluster_centroids = cluster_centroids[0:1]
            return self.get_grad_mask(x, cluster_centroids, th=th)

        if self.fix_encoder:
            with torch.no_grad():
                y, attn = self.encoder.forward_feat_attn(x)
        else:
            y, attn = self.encoder.forward_feat_attn(x)

        masks_cls, cls_embs = self.decoder(y[:, 1:], ret_cls_emb=True)
        masks_cls = masks_cls.transpose(1, 2).view(N, N_cls_fgbg, h, w)

        if return_loss:
            losses = dict()
            loss_cls_emb = cls_emb_loss(cls_embs)

            if bootstrapping:
                with torch.no_grad():
                    self.decoder_teacher.eval()
                    self.encoder_teacher.eval()
                    y_teacher, _ = self.encoder_teacher.forward_feat_attn(x)
                    pseudo_labels_teacher = self.decoder_teacher(y_teacher[:, 1:])
                    pseudo_labels_teacher = pseudo_labels_teacher.transpose(1, 2).view(N, N_cls_fgbg, h, w)
            else:
                pseudo_labels_teacher = None
            if not pseudo_labels.shape[-2:] == (h, w):
                pseudo_labels = F.interpolate(pseudo_labels, size=(h, w), mode='bilinear', align_corners=False)
            loss_cat, loss_uncertainty, bootstrapped_pseudo_labels = \
                thingstuff_bootstrap_loss(masks_cls, pseudo_label=pseudo_labels,
                                          pseudo_label_teacher=pseudo_labels_teacher, epoch=epoch)


            losses['loss_cat'] = loss_cat
            losses['loss_uncertainty'] = loss_uncertainty
            losses['loss_cls_emb'] = loss_cls_emb

            return losses

        return masks_cls.view(N, N_cls_fgbg, h, w)
