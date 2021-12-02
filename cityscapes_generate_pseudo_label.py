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

from utils.kmeans import train_kmeans

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms

from models.segmenter.segmenter_thingstuff import Segmenter as Model


from dataloaders import transforms_uss_thingstuff
from dataloaders.cityscapes_trainid_map import cityscape_id_idx_map
from utils.misc import AverageMeter



ex = Experiment('cityscapes')

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

ex.add_config('./configs/cityscapes.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True


def load_train_dataset_cityscapes(cfg, split='train', sample_range=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.NormInput(),
        transforms_uss_thingstuff.ToTensor(),
        transforms_uss_thingstuff.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize*2), img_only=False),
    ])

    dataset = Cityscapes(transform=train_transform,
                         split=split,
                         dataset_root_dir=cfg.dataset.root_dir_cityscapes,
                         sample_range=sample_range)
    return dataset


class Cityscapes(Dataset):
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

        city_names = os.listdir(os.path.join(self.dataset_root_dir, 'leftImg8bit', self.split))

        samples_list = []
        for city_name in city_names:
            samples = os.listdir(os.path.join(self.dataset_root_dir, 'leftImg8bit', str(self.split), str(city_name)))
            for sample in samples:
                sample_name_prefix = '_'.join(sample.split('_')[:3])
                img = f'leftImg8bit/{self.split}/{city_name}/{sample_name_prefix}_leftImg8bit.png'
                label = f'gtFine/{self.split}/{city_name}/{sample_name_prefix}_gtFine_labelIds.png'
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
        _cityscape_id_idx_map = np.vectorize(lambda x: cityscape_id_idx_map[x])
        labels = _cityscape_id_idx_map(labels)

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
def slide_window_cls_feat_clustering(cfg, feat_save_dir, dataset, _log):
    N_things = cfg.model.decoder.n_things
    N_stuff = cfg.model.decoder.n_stuff
    N_cls = N_things + N_stuff
    h, w = cfg.dataset.resize // cfg.model.encoder.patch_size, (cfg.dataset.resize * 2) // cfg.model.encoder.patch_size
    cluster_centroids_save_path = os.path.join(feat_save_dir, f'cluster_centroids_slide_window_{N_cls}_{h}x{w}')
    if not os.path.exists(cluster_centroids_save_path):
        samples_list = dataset.samples_list
        cls_feature_all = []
        for i, sample in enumerate(samples_list):
            file_name = '_'.join(sample['images'].split('/')[-1].split('.')[0].split('_')[:3])
            cls_feat_save_path = os.path.join(feat_save_dir, f"{file_name}_cls_feat_{N_cls}_{int(min(h, w))}x{int(min(h, w))}")
            if os.path.exists(cls_feat_save_path):
                dictfile = open(cls_feat_save_path, 'rb')
                cls_feat_k = pickle.load(dictfile)
                cls_feature_all.append(torch.from_numpy(cls_feat_k))
                _log.info(f"id {i+1}/{len(samples_list)} fg feature loaded from {cls_feat_save_path}")
            else:
                _log.info(f"{i+1}/{len(samples_list)} fg feature do not found at {cls_feat_save_path}")
        cls_feature_all = torch.cat(cls_feature_all)
        cluster_centroids, labels = train_kmeans(cls_feature_all, N_cls, metric='l2')

        dictfile = open(cluster_centroids_save_path, 'wb')
        pickle.dump(cluster_centroids.cpu().numpy(), dictfile)
        dictfile.close()
        _log.info(f"clustering done, save cluster_centroids to {cluster_centroids_save_path}")
    else:
        cluster_centroids = pickle.load(open(cluster_centroids_save_path, 'rb'))
        cluster_centroids = torch.from_numpy(cluster_centroids)
        _log.info(f"load cluster_centroids from {cluster_centroids_save_path}")

    return cluster_centroids


@ex.capture
def generate_pseudo_label_cityscapes(cfg, model, data_loader, cluster_centroids, pseudo_label_save_dir, device, _log):
    batch_time = AverageMeter()
    tic = time.time()
    model.train()
    N_things = cfg.model.decoder.n_things
    N_stuff = cfg.model.decoder.n_stuff
    N_cls = N_things + N_stuff
    assert cluster_centroids.shape[0] == N_cls
    for index, sample in enumerate(data_loader):
        images = sample['images'].float().to(device)   # image, normalized
        sample_names = sample['meta']['sample_name']

        pseudo_labels = model(images,
                              cluster_centroids=cluster_centroids[None].expand(images.shape[0], -1, -1).to(device))
        N_cls, h, w = pseudo_labels.shape[1:]
        assert pseudo_labels.shape[1:] == (N_cls, h, w)
        for sample_name, pseudo_label in zip(sample_names, pseudo_labels):

            pseudo_label_fg_save_path = os.path.join(pseudo_label_save_dir, sample_name+f"_{N_cls}_{h}x{w}")
            pseudo_label = pseudo_label.cpu().numpy()
            dictfile = open(pseudo_label_fg_save_path, 'wb')
            pickle.dump(pseudo_label, dictfile)
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

    feature_bank_dir = os.path.join(cfg.dataset.root_dir_cls_feat, f'cls_feat_cityscapes')
    pseudo_label_save_dir = os.path.join(cfg.dataset.root_dir_pseudo_label, f'pseudo_label_cityscapes')

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
    train_dataset_cityscapes = load_train_dataset_cityscapes(cfg, split=cfg.dataset.split, sample_range=sample_range)

    cluster_centroids = slide_window_cls_feat_clustering(cfg, feature_bank_dir, train_dataset_cityscapes)
    sampler = None
    train_loader = DataLoader(train_dataset_cityscapes, batch_size=cfg.dataset.train_batch_size,
                              shuffle=False,
                              num_workers=cfg.dataset.num_workers,
                              pin_memory=True, drop_last=False, sampler=sampler)
    generate_pseudo_label_cityscapes(cfg, model, train_loader, cluster_centroids, pseudo_label_save_dir, device)
