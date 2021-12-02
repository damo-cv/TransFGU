'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/janghyuncho/PiCIE/blob/master/utils.py
'''
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)  # Exclude unlabelled data.
    hist = np.bincount(n_class * label_true[mask] + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist

def get_result_metrics(histogram):
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn)
    opc = np.sum(tp) / np.sum(histogram)

    result = {"iou": iou,
             "mean_iou": np.nanmean(iou),
             "precision_per_class (per class accuracy)": prc,
             "mean_precision (class-avg accuracy)": np.nanmean(prc),
             "overall_precision (pixel accuracy)": opc}

    result = {k: 100*v for k, v in result.items()}

    return result
