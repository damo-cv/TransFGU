'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import os
import time
import random

import numpy as np
from sacred import Experiment
import logging
from easydict import EasyDict as edict
from PIL import Image
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms

from models.segmenter.segmenter_thingstuff import Segmenter as Model
from models.utils.bbox_anchors import generate_box_anchors_by_scale

from dataloaders import transforms_uss_thingstuff
from dataloaders.lip_id_idx_map import lip_id_idx_coarse_16_map
from utils.misc import AverageMeter

ex = Experiment('lip')

def create_basic_stream_logger(format):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

ex.logger = create_basic_stream_logger('%(levelname)s - %(name)s - %(message)s')

ex.add_config('./configs/lip.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True


def load_dataset_lip(cfg, split='val', sample_range=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.NormInput(),
        transforms_uss_thingstuff.ToTensor(),
        transforms_uss_thingstuff.ResizeTensor(size=(cfg.dataset.resize*2, cfg.dataset.resize), img_only=False),
    ])

    dataset = LIP(transform=train_transform,
                  split=split,
                  dataset_root_dir=cfg.dataset.root_dir_lip,
                  sample_range=sample_range)
    return dataset


class LIP(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 sample_range=None):
        assert split in ['train', 'val']
        self.split = split
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples

        self.samples = []
        with open(os.path.join(self.dataset_root_dir, self.split + '_id.txt')) as f:
            samples_tmp = f.readlines()
        samples_tmp = list(map(lambda elem: elem.strip(), samples_tmp))
        self.samples.extend(samples_tmp)


        samples_list = []
        for sample in self.samples:
            img = f'{self.split}_images/{sample}.jpg'
            label = f'{self.split}_segmentations/{sample}.png'
            sample = dict()
            sample['images'] = img
            sample['labels'] = label
            samples_list.append(sample)

        if self.num_samples is not None:
            samples_list = samples_list[:self.num_samples]
        elif sample_range is not None:
            assert isinstance(sample_range, tuple)
            samples_list = samples_list[sample_range[0]:sample_range[1]]
        self.samples_list = samples_list


    def __len__(self):
        return len(self.samples_list)


    def __getitem__(self, idx):

        images = self.samples_list[idx]['images']
        labels = self.samples_list[idx]["labels"]
        sample_name = '_'.join(images.split('/')[-1].split('.')[0].split('_')[:3])
        # images
        images = np.array(Image.open(os.path.join(self.dataset_root_dir, images)).convert('RGB'))
        labels = np.array(Image.open(os.path.join(self.dataset_root_dir, labels)), dtype=np.int8)
        _lip_id_idx_map = np.vectorize(lambda x: lip_id_idx_coarse_16_map[x])
        labels = _lip_id_idx_map(labels)

        sample_ = dict()
        sample_['img'] = images
        sample_['label_cat'] = labels
        sample_['meta'] = {'sample_name': sample_name}

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()

        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']
        sample['meta'] = sample_['meta']

        return sample



@ex.capture
def crop_image_slide_window(images, anchors, h, w, H_crop, W_crop, _log):
    '''

    '''
    N, _, H, W = images.shape
    image_crops_list = []
    N_anchors = anchors.shape[0]
    anchors = F.interpolate(anchors.view(N_anchors, h, w)[None], (H, W))[0]
    for image in images:
        image_crops = []
        for i, anchor in enumerate(anchors):
            xy = anchor.nonzero()
            assert len(xy) > 0
            x0, x1, y0, y1 = xy[:, 1].min(), xy[:, 1].max(), xy[:, 0].min(), xy[:, 0].max()
            image_crops.append(F.interpolate(image[:, y0:y1, x0:x1][None], (H_crop, W_crop), mode='bilinear'))
        assert len(image_crops) > 0
        image_crops = torch.cat(image_crops)
        image_crops_list.append(image_crops)
    return image_crops_list


@ex.capture
def extract_features_and_save(model, model_val, image_crops, batch_size_crops, embed_dim, device, H, W, feat_save_dir, sample_name, _log):
    N_crops, _, H_crop, W_crop = image_crops.shape
    N_batched_crops = (N_crops // batch_size_crops) * batch_size_crops
    batch_image_crops = image_crops[:N_batched_crops]
    N_rest = N_crops - N_batched_crops
    batch_image_crops = batch_image_crops.view(N_crops // batch_size_crops, batch_size_crops, 3, H // 2, W // 2)
    y_cls = torch.zeros(N_crops, embed_dim).to(device)
    for i, batch_image_crops_ in enumerate(batch_image_crops):
        y_cls[i * batch_size_crops: (i + 1) * batch_size_crops] = model(batch_image_crops_.to(device))
    if N_rest > 0:
        for i in range(N_rest):
            y_cls[N_batched_crops + i: N_batched_crops + i + 1] = \
                model_val(image_crops[N_batched_crops + i: N_batched_crops + i + 1].to(device))

    feat_save_path = os.path.join(feat_save_dir, sample_name)
    cls_feat = y_cls.cpu().numpy()
    dictfile = open(feat_save_path, 'wb')
    pickle.dump(cls_feat, dictfile)
    dictfile.close()


@ex.capture
def crop_and_extract_features_coco_slide_window(cfg, model, model_val, data_loader, device, feat_save_dir, _log):
    batch_time = AverageMeter()
    tic = time.time()
    model.eval()

    batch_size_crops = cfg.dataset.train_batch_size // 2
    anchors = generate_box_anchors_by_scale(cfg.dataset.resize*2 // cfg.model.encoder.patch_size,
                                            (cfg.dataset.resize) // cfg.model.encoder.patch_size,
                                            scales=(0.5, 0.4, 0.3, 0.2))
    anchors = anchors.to(device)
    for index, sample in enumerate(data_loader):

        images = sample['images'].float().to(device)   # image, normalized
        meta = sample['meta']
        sample_names = sample['meta']['sample_name']
        N, _, H, W = images.shape
        N_cls = cfg.model.decoder.n_things + cfg.model.decoder.n_stuff
        h, w = H // cfg.model.encoder.patch_size, W // cfg.model.encoder.patch_size
        H_crop = W_crop = int(min(H // 2, W // 2))
        embed_dim = model_val.embed_dim
        for image, sample_name in zip(images, sample_names):
            image_crops = crop_image_slide_window(image[None], anchors, h, w, H_crop=H_crop, W_crop=W_crop)
            image_crops = torch.cat(image_crops)
            N_crops = image_crops.shape[0]
            h_crop, w_crop = H_crop // cfg.model.encoder.patch_size, W_crop // cfg.model.encoder.patch_size
            N_batched_crops = (N_crops // batch_size_crops) * batch_size_crops
            batch_image_crops = image_crops[:N_batched_crops]
            N_rest = N_crops - N_batched_crops
            batch_image_crops = batch_image_crops.view(N_crops // batch_size_crops, batch_size_crops, 3, H_crop, W_crop)
            y_cls = torch.zeros(N_crops, embed_dim).to(device)
            for i, batch_image_crops_ in enumerate(batch_image_crops):
                y_cls[i * batch_size_crops : (i+1) * batch_size_crops] = model(batch_image_crops_.to(device))
            if N_rest > 0:
                for i in range(N_rest):
                    y_cls[N_batched_crops + i : N_batched_crops + i + 1] = \
                        model_val(image_crops[N_batched_crops + i : N_batched_crops + i + 1].to(device))

            feat_save_path = os.path.join(feat_save_dir, sample_name+f"_cls_feat_{h_crop}x{w_crop}")
            cls_feat = y_cls.cpu().numpy()
            dictfile = open(feat_save_path, 'wb')
            pickle.dump(cls_feat, dictfile)
            dictfile.close()

        # update time
        batch_time.update(time.time() - tic)
        tic = time.time()

        _log.info(f"[{index + 1:4d}/{len(data_loader):4d}]\t"
                  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f})")

    _log.info(f"done")


@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)
    cfg['dataset']['resize'] = cfg['dataset']['resize'] * 2
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_bank_dir = os.path.join(cfg.dataset.root_dir_cls_feat, f'cls_feat_lip')
    os.makedirs(feature_bank_dir, exist_ok=True)

    with torch.no_grad():
        # Network Builders
        model = Model(cfg.model).encoder
        if torch.cuda.is_available():
            model.cuda()
            if torch.cuda.device_count() > 1:
                _log.info(f"using {torch.cuda.device_count()} gpus")
                model = torch.nn.DataParallel(model)

        sample_range = (cfg.dataset.range_start, cfg.dataset.range_end) if cfg.dataset.range_start > -1 and cfg.dataset.range_end > -1 else None
        dataset_lip = load_dataset_lip(cfg, split=cfg.dataset.split, sample_range=sample_range)
        loader_lip = DataLoader(dataset_lip, batch_size=1, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

        model_val = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            crop_and_extract_features_coco_slide_window(cfg, model, model_val, loader_lip, device, feature_bank_dir, )