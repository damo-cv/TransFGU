'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import torch

def cls_emb_loss(cls_emb, dist='cos'):
    '''
    cls_emb:  N x N_cls x N_dim, [0, 1], normalized
    '''
    N, N_cls, N_dim = cls_emb.shape
    if dist == 'cos':
        dist = cls_emb @ cls_emb.transpose(1, 2)
        dist = dist.triu(1)
        dist_ = dist.masked_select(torch.ones_like(dist).triu(1).bool()).view(N, -1)
        loss = 1 + dist_
    else:
        raise NotImplementedError

    return loss