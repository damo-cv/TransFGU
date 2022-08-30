'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import torch
import random
import torch.nn.functional as F

alpha_list = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]


def thingstuff_bootstrap_loss(masks, pseudo_label=None,
                              pseudo_label_teacher=None, epoch=None):
    '''
    mask:  N x N_cls x hw, [0, 1], includes bg
    label: N x hw, binary
    '''
    N, N_cls_fgbg, h, w = masks.shape

    pseudo_label_blur = pseudo_label
    pseudo_label_norm = torch.stack([(label - label.min()) / (label.max() - label.min()) for label in pseudo_label_blur])

    if pseudo_label_teacher is not None:
        w_1 = 0.5
        w_2 = 0.5
        pseudo_label_teacher_blur = pseudo_label_teacher
        pseudo_label_teacher_norm = pseudo_label_teacher_blur.softmax(1)
        pseudo_label_norm = w_1 * pseudo_label_teacher_norm + w_2 * pseudo_label_norm

    bootstrapped_pseudo_labels = pseudo_label_norm.max(1)[1]

    if pseudo_label_teacher is not None:

        alpha = alpha_list[-1] if not (epoch < len(alpha_list)) else alpha_list[epoch]

        # negative sample: random shuffle
        idx_list = list(range(N_cls_fgbg))
        random.shuffle(idx_list)
        bootstrapped_pseudo_labels_negative = pseudo_label_norm[:, idx_list].max(1)[1]

        loss_cat = F.cross_entropy(masks, bootstrapped_pseudo_labels, reduce=False)\
                   - alpha * F.cross_entropy(masks, bootstrapped_pseudo_labels_negative, reduce=False)
    else:
        loss_cat = F.cross_entropy(masks, bootstrapped_pseudo_labels, reduce=False)

    mask_topk = masks.view(N, N_cls_fgbg, h, w).softmax(1).topk(N_cls_fgbg, dim=1)[0]
    loss_uncertainty = 1 - (mask_topk[:, 0] - mask_topk[:, 1])

    return loss_cat, loss_uncertainty, bootstrapped_pseudo_labels


def things_bootstrap_loss(masks, pseudo_label=None, pseudo_label_teacher=None,
                          mask_bin_th=0.1, epoch=None):
    '''
    mask:  N x N_cls x hw, [0, 1], includes bg
    label: N x hw, binary
    '''
    N, N_cls_fgbg, h, w = masks.shape

    pseudo_label_blur = pseudo_label
    pseudo_label_norm = torch.stack([(label - label.min()) / (label.max() - label.min()) for label in pseudo_label_blur])
    fg_area = (pseudo_label_norm > mask_bin_th).sum(1, keepdim=True)
    mask_bg = F.relu_((0.1 - pseudo_label_norm.max(1, keepdim=True)[0]))
    pseudo_label_norm = torch.cat([mask_bg, pseudo_label_norm], 1)


    if pseudo_label_teacher is not None:
        w_1 = 0.5
        w_2 = 0.5
        pseudo_label_teacher_blur = pseudo_label_teacher
        pseudo_label_teacher_norm = pseudo_label_teacher_blur.softmax(1)
        pseudo_label_norm = w_1 * pseudo_label_teacher_norm + w_2 * pseudo_label_norm

    bootstrapped_pseudo_labels = pseudo_label_norm.max(1)[1]  # standard pseudo label

    if pseudo_label_teacher is not None:

        alpha = alpha_list[-1] if not (epoch < len(alpha_list)) else alpha_list[epoch]

        # negative sample: random shuffle
        idx_list = list(range(N_cls_fgbg))
        random.shuffle(idx_list)
        bootstrapped_pseudo_labels_negative = pseudo_label_norm[:, idx_list].max(1)[1]

        loss_cat = F.cross_entropy(masks, bootstrapped_pseudo_labels, reduce=False)\
                   - alpha * F.cross_entropy(masks, bootstrapped_pseudo_labels_negative, reduce=False)
    else:
        loss_cat = F.cross_entropy(masks, bootstrapped_pseudo_labels, reduce=False)

    mask_topk = masks[:, 1:].view(N, N_cls_fgbg - 1, h, w).softmax(1).topk(N_cls_fgbg - 1, dim=1)[0]
    loss_uncertainty = 1 - (mask_topk[:, 0:1].masked_select(fg_area.bool()) -
                            mask_topk[:, 1:2].masked_select(fg_area.bool())).mean()

    return loss_cat, loss_uncertainty, bootstrapped_pseudo_labels

