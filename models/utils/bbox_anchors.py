'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import torch


def generate_box(h, w, h_center, w_center, h_bbox, w_bbox):
    '''
    h, w:   size of map

    '''
    y, x = torch.meshgrid(torch.Tensor(list(range(h))), torch.Tensor(list(range(w))))
    dist_h = (y - h_center).abs()
    dist_w = (x - w_center).abs()
    dist_h[dist_h > h_bbox/2] = -1
    dist_w[dist_w > w_bbox/2] = -1
    mask = torch.zeros_like(dist_h)
    mask[y.flatten().long(), x.flatten().long()] = (((dist_h.flatten() >= 0) & (dist_w.flatten() >= 0))).float()
    return mask

def generate_box_anchors(h, w, h_bbox, w_bbox, interval):
    '''
    h_bbox: odd number
    w_bbox: odd number
    '''

    # assert not h_bbox % 2 == 0 and not w_bbox % 2 == 0
    y, x = torch.meshgrid(torch.Tensor(list(range(h))), torch.Tensor(list(range(w))))

    anchors = []
    for y_, x_ in zip(y.flatten(), x.flatten()):
        if (((x_ - w_bbox/2) >= 0 and (y_ - h_bbox/2) >= 0) and ((x_ + w_bbox/2) < w and (y_ + h_bbox/2) < h)):
            if (x_ % interval == 0 and y_ % interval == 0) \
                    or ((y_ % interval == 0) and (((x_ - w_bbox/2) == 0) or ((x_ + w_bbox/2) == (w-1)))) \
                    or ((x_ % interval == 0) and (((y_ - h_bbox/2) == 0) or ((y_ + h_bbox/2) == (h-1))))\
                    or (((x_ + w_bbox/2) == (w-1)) and ((y_ + h_bbox/2) == (h-1))):

                anchors.append(generate_box(h, w, y_, x_, h_bbox, w_bbox).flatten())

    return anchors


def generate_box_anchors_by_scale(h, w, scales=(0.4, 0.3, 0.2, 0.1)):
    anchors = []
    for scale in scales:
        anchors += generate_box_anchors(h, w,
                                        int(min(h * scale, w * scale)),
                                        int(min(h * scale, w * scale)),
                                        int(min(h * scale, w * scale)) / 2)
    assert len(anchors) > 0
    anchors = torch.stack(anchors, 0)
    return anchors