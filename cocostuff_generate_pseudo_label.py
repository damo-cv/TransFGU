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

from utils.kmeans import train_kmeans_faiss as train_kmeans

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms

from models.segmenter.segmenter_thingstuff import Segmenter as Model


from dataloaders import transforms_uss_thingstuff
from dataloaders.coco_id_idx_map import coco_stuff_id_idx_map
from utils.misc import AverageMeter, load_network_checkpoint

from pycocotools.coco import COCO


ex = Experiment('uss_thingstuff')

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

ex.add_config('./configs/cocostuff.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True


def load_train_dataset_coco(cfg, split='train', sample_range=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.Resize(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
        transforms_uss_thingstuff.NormInput(),
        transforms_uss_thingstuff.ToTensor(),
    ])

    dataset = MSCOCO17(transform=train_transform,
                         split=split,
                         dataset_root_dir=cfg.dataset.root_dir_mscoco,
                         sample_range=sample_range,
                         anno_type='instance&object',
                         orientation=0)
    return dataset


class MSCOCO17(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 sample_range=None,
                 orientation=0,
                 anno_type='instances'
                 ):
        assert split in ['train', 'val']
        self.split = 'train2017' if split == 'train' else 'val2017'
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples
        assert anno_type in ['instances', 'object', 'instance&object']
        self.anno_type = anno_type

        self.JPEGPath = f"{self.dataset_root_dir}/images/{self.split}"
        self.PNGPath = f"{self.dataset_root_dir}/annotations/{self.split}"
        self.annFile = f"{self.dataset_root_dir}/annotations/instances_{self.split}.json"
        self.coco = COCO(self.annFile)
        all_ids = self.coco.imgToAnns.keys()

        samples_list_1 = []
        samples_list_2 = []
        for id in all_ids:

            img_meta = self.coco.loadImgs(id)
            assert len(img_meta) == 1
            H, W = img_meta[0]['height'], img_meta[0]['width']

            if H < W:
                samples_list_1.append(id)
            else:
                samples_list_2.append(id)

        if orientation == 0:
            samples_list = samples_list_1 + samples_list_2
        elif orientation == 1:
            samples_list = samples_list_1
        elif orientation == 2:
            samples_list = samples_list_2
        else:
            raise NotImplementedError

        if self.num_samples is not None:
            samples_list = samples_list[:self.num_samples]
        elif sample_range is not None:
            assert isinstance(sample_range, tuple)
            samples_list = samples_list[sample_range[0]:sample_range[1]]
        self.samples_list = samples_list


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        id = self.samples_list[idx]
        img_meta = self.coco.loadImgs(id)
        assert len(img_meta) == 1
        img_meta = img_meta[0]

        # image
        image = np.array(Image.open(f"{self.JPEGPath}/{img_meta['file_name']}").convert('RGB'))
        label_cat = np.array(Image.open(f"{self.PNGPath}/{img_meta['file_name'].replace('jpg', 'png')}"))

        _coco_id_idx_map = np.vectorize(lambda x: coco_stuff_id_idx_map[x])
        label_cat = _coco_id_idx_map(label_cat)
        sample_ = dict()
        sample_['img'] = image
        sample_['label_cat'] = label_cat
        sample_['meta'] = {'sample_name': img_meta['file_name'].split('.')[0]}

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()

        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']
        sample['meta'] = sample_['meta']

        return sample


@ex.capture
def slide_window_cls_feat_clustering(cfg, feat_save_dir, dataset, device, _log):
    N_things = cfg.model.decoder.n_things
    N_stuff = cfg.model.decoder.n_stuff
    N_cls = N_things + N_stuff
    h = w = cfg.dataset.resize // cfg.model.encoder.patch_size
    cluster_centroids_fg_save_path = os.path.join(feat_save_dir, f'cluster_centroids_slide_window_fg_{N_things}_{h}x{w}')
    cluster_centroids_bg_save_path = os.path.join(feat_save_dir, f'cluster_centroids_slide_window_bg_{N_stuff}_{h}x{w}')
    if not os.path.exists(cluster_centroids_fg_save_path) or not os.path.exists(cluster_centroids_bg_save_path):
        samples_list = dataset.samples_list
        cls_feature_fg_all = []
        cls_feature_bg_all = []
        for i, id in enumerate(samples_list):
            img_meta = dataset.coco.loadImgs(id)[0]

            cls_feat_fg_save_path = os.path.join(feat_save_dir, f"{img_meta['file_name'].split('.')[0]}_cls_feat_fg_{h}x{w}")
            cls_feat_bg_save_path = os.path.join(feat_save_dir, f"{img_meta['file_name'].split('.')[0]}_cls_feat_bg_{h}x{w}")
            if os.path.exists(cls_feat_fg_save_path):
                dictfile = open(cls_feat_fg_save_path, 'rb')
                cls_feat_k = pickle.load(dictfile)
                cls_feature_fg_all.append(torch.from_numpy(cls_feat_k))
                _log.info(f"id {i+1}/{len(samples_list)} fg feature loaded from {cls_feat_fg_save_path}")
            else:
                _log.info(f"{i+1}/{len(samples_list)} fg feature do not found at {cls_feat_fg_save_path}")
            if os.path.exists(cls_feat_bg_save_path):
                dictfile = open(cls_feat_bg_save_path, 'rb')
                cls_feat_k = pickle.load(dictfile)
                cls_feature_bg_all.append(torch.from_numpy(cls_feat_k))
                _log.info(f"id {i+1}/{len(samples_list)} fg feature loaded from {cls_feat_bg_save_path}")
            else:
                _log.info(f"{i+1}/{len(samples_list)} fg feature do not found at {cls_feat_bg_save_path}")
        cls_feature_fg_all = torch.cat(cls_feature_fg_all).to(device)
        cls_feature_bg_all = torch.cat(cls_feature_bg_all).to(device)
        cluster_centroids_fg, labels_fg = train_kmeans(cls_feature_fg_all, N_things, metric='l2')
        cluster_centroids_bg, labels_bg = train_kmeans(cls_feature_bg_all, N_stuff, metric='l2')

        dictfile = open(cluster_centroids_fg_save_path, 'wb')
        pickle.dump(cluster_centroids_fg.cpu().numpy(), dictfile)
        dictfile.close()
        _log.info(f"clustering done, save cluster_centroids to {cluster_centroids_fg_save_path}")
        dictfile = open(cluster_centroids_bg_save_path, 'wb')
        pickle.dump(cluster_centroids_bg.cpu().numpy(), dictfile)
        dictfile.close()
        _log.info(f"clustering done, save cluster_centroids to {cluster_centroids_bg_save_path}")
    else:
        cluster_centroids_fg = pickle.load(open(cluster_centroids_fg_save_path, 'rb'))
        cluster_centroids_fg = torch.from_numpy(cluster_centroids_fg)
        _log.info(f"load cluster_centroids_fg from {cluster_centroids_fg_save_path}")
        cluster_centroids_bg = pickle.load(open(cluster_centroids_bg_save_path, 'rb'))
        cluster_centroids_bg = torch.from_numpy(cluster_centroids_bg)
        _log.info(f"load cluster_centroids_bg from {cluster_centroids_bg_save_path}")

    return cluster_centroids_fg, cluster_centroids_bg


@ex.capture
def generate_pseudo_label_coco(cfg, model, data_loader, cluster_centroids_fg, cluster_centroids_bg, pseudo_label_save_dir, device, _log):
    batch_time = AverageMeter()
    tic = time.time()
    model.train()
    N_things = cfg.model.decoder.n_things
    N_stuff = cfg.model.decoder.n_stuff
    N_cls = N_things + N_stuff
    cluster_centroids = torch.cat([cluster_centroids_fg, cluster_centroids_bg], 0)
    assert cluster_centroids.shape[0] == N_cls
    for index, sample in enumerate(data_loader):
        images = sample['images'].float().to(device)   # image, normalized
        sample_names = sample['meta']['sample_name']

        pseudo_labels = model(images,
                              cluster_centroids=cluster_centroids[None].expand(images.shape[0], -1, -1).to(device))
        N_cls, h, w = pseudo_labels.shape[1:]
        pseudo_labels_fg = pseudo_labels[:, :N_things]
        pseudo_labels_bg = pseudo_labels[:, N_things:]
        assert pseudo_labels_fg.shape[1:] == (N_things, h, w)
        assert pseudo_labels_bg.shape[1:] == (N_stuff, h, w)
        assert h == w
        for sample_name, pseudo_label_fg, pseudo_label_bg in zip(sample_names, pseudo_labels_fg, pseudo_labels_bg):

            pseudo_label_fg_save_path = os.path.join(pseudo_label_save_dir, sample_name + f"_fg_{N_things}_{h}x{w}")
            pseudo_label_fg = pseudo_label_fg.cpu().numpy()
            dictfile = open(pseudo_label_fg_save_path, 'wb')
            pickle.dump(pseudo_label_fg, dictfile)
            dictfile.close()

            pseudo_label_bg_save_path = os.path.join(pseudo_label_save_dir, sample_name + f"_bg_{N_stuff}_{h}x{w}")
            pseudo_label_bg = pseudo_label_bg.cpu().numpy()
            dictfile = open(pseudo_label_bg_save_path, 'wb')
            pickle.dump(pseudo_label_bg, dictfile)
            dictfile.close()

        # update time
        batch_time.update(time.time() - tic)
        tic = time.time()

        _log.info(f"[{index + 1:4d}/{len(data_loader):4d}]\t"
                  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                  f"N_cls: {N_cls}\tsize:{h}x{w}")

    _log.info(f"Done")


@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_bank_dir = os.path.join(cfg.dataset.root_dir_cls_feat, f'cls_feat_cocostuff')
    pseudo_label_save_dir = os.path.join(cfg.dataset.root_dir_pseudo_label, f'pseudo_label_cocostuff_{cfg.model.decoder.n_things+cfg.model.decoder.n_stuff}')

    assert os.path.exists(feature_bank_dir)
    os.makedirs(pseudo_label_save_dir, exist_ok=True)

    model = Model(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
        if torch.cuda.device_count() > 1:
            _log.info(f"using {torch.cuda.device_count()} gpus")
            model = torch.nn.DataParallel(model)

    if cfg.dataset.range_start > -1 and cfg.dataset.range_end > -1:
        sample_range = (cfg.dataset.range_start, cfg.dataset.range_end)
    else:
        sample_range = None
    train_dataset_coco = load_train_dataset_coco(cfg, split=cfg.dataset.split, sample_range=sample_range)

    cluster_centroids_fg, cluster_centroids_bg = slide_window_cls_feat_clustering(cfg, feature_bank_dir, train_dataset_coco, device)
    sampler = None
    train_loader = DataLoader(train_dataset_coco, batch_size=cfg.dataset.train_batch_size,
                              shuffle=False,
                              num_workers=cfg.dataset.num_workers,
                              pin_memory=True, drop_last=False, sampler=sampler)
    generate_pseudo_label_coco(cfg, model, train_loader, cluster_centroids_fg, cluster_centroids_bg, pseudo_label_save_dir, device)
