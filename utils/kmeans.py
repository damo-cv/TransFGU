'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/facebookresearch/faiss/blob/main/tutorial/python/1-Flat.py
'''
import faiss
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

import torch


def train_kmeans(x, k, niter=100, metric='l2', min_points_per_centroid=None, gpu_id=0, seed=1, verbose=False, use_mini_batch=True):
    '''
    Runs kmeans on one or several GPUs
    :param x:           tensor, N x d, float
    :param k:           number of cluster centroid
    :param niter:
    :param metric:      l2 or ip (for inner product)
    :param gpu_id:
    :param seed:        integer, greater than 0
    :param verbose:
    :return:            cluster centroid with k x d, indice with N x 1
    '''
    metric_list = ['l2', 'ip', 'cos']
    assert metric in metric_list
    d = x.shape[1]
    if metric == 'ip' or metric == 'cos':
        x_ = x / x.norm(dim=1, keepdim=True)
    else:
        x_ = x
    if use_mini_batch:
        cluster = MiniBatchKMeans(n_clusters=k, max_iter=niter, random_state=0).fit(x_.cpu().numpy())
    else:
        cluster = KMeans(n_clusters=k, max_iter=niter, random_state=0).fit(x_.cpu().numpy())
    label = cluster.labels_
    cluster_centers = torch.zeros(k, d)

    for n in range(k):
        cluster_centers[n] = x[label == n].mean(0)

    # dist_to_center = 0
    # for n in range(k):
    #     dist_to_center += ((x[label == n] - cluster_centers[n][None]) ** 2).sum(-1).sqrt().mean()
    # dist_to_center /= k
    # print(f"k:{k}, dist to center:{dist_to_center}")

    return torch.Tensor(cluster_centers), torch.Tensor(cluster.labels_)


def train_kmeans_faiss(x, k, niter=100, metric='l2', min_points_per_centroid=None, gpu_id=0, seed=1, verbose=False):
    '''
    Runs kmeans on one or several GPUs
    :param x:           Tensor, N x d, float
    :param k:           number of cluster centroid
    :param niter:
    :param metric:      l2 or ip (for inner product)
    :param gpu_id:
    :param seed:        integer, greater than 0
    :param verbose:
    :return:            cluster centroid with k x d, indice with N x 1
    '''
    metric_list = ['l2', 'ip', 'cos']
    assert metric in metric_list
    d = x.shape[1]
    device = x.device
    clus = faiss.Clustering(d, k)
    clus.seed = int(np.array(seed)) if seed is not None else np.random.randint(2021)
    clus.verbose = verbose
    clus.niter = niter

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000
    if min_points_per_centroid is not None:
        clus.min_points_per_centroid = min_points_per_centroid


    if str(device) == 'cpu':
        if metric == 'l2':
            index = faiss.IndexFlatL2(d)
        elif metric == 'ip' or metric == 'cos':
            index = faiss.IndexFlatIP(d)
        else:
            raise NotImplementedError(f"metric must be in the range of {metric_list}")
    else:
        res = faiss.StandardGpuResources()
        flat_config = []
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id
        flat_config.append(cfg)

        if metric == 'l2':
            index = faiss.GpuIndexFlatL2(res, d, flat_config[0])
        elif metric == 'ip' or metric == 'cos':
            index = faiss.GpuIndexFlatIP(res, d, flat_config[0])
        else:
            raise NotImplementedError(f"metric must be in the range of {metric_list}")

    # perform the training
    input = np.ascontiguousarray(x.detach().cpu().numpy())
    clus.train(x=input, index=index)
    # centroids = faiss.vector_float_to_array(clus.centroids)
    D, I = index.search(input, 1)
    # obj = faiss.vector_float_to_array(clus.obj)
    # print("final objective: %.4g" % obj[-1])


    cluster_centers = torch.zeros(k, d)
    for n in range(k):
        cluster_centers[n] = x[I.squeeze(1) == n].mean(0)

    return torch.Tensor(cluster_centers).to(device), torch.Tensor(I).squeeze(1).to(device)
