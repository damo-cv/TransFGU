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
from utils.misc import AverageMeter, load_network_checkpoint



ex = Experiment('pascalvoc')

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

ex.add_config('./configs/pascalvoc.yaml')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True


def load_train_dataset_pascalvoc(cfg, split='train', sample_range=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.NormInput(),
        transforms_uss_thingstuff.ToTensor(),
        transforms_uss_thingstuff.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])

    dataset = PascalVOC(transform=train_transform,
                         split=split,
                         dataset_root_dir=cfg.dataset.root_dir_pascalvoc,
                         sample_range=sample_range,
                         orientation=0)
    return dataset


class PascalVOC(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 transform=None,
                 num_samples=None,
                 sample_range=None,
                 orientation=0):

        assert split in ['train', 'val']
        self.split = split
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples
        self.anno_type = 'SegmentationClass'

        self.samples = []
        with open(os.path.join(self.dataset_root_dir, 'ImageSets', 'Segmentation', self.split + '.txt')) as f:
            samples_tmp = f.readlines()
        samples_tmp = list(map(lambda elem: elem.strip(), samples_tmp))
        self.samples.extend(samples_tmp)


        samples_list_1 = []
        samples_list_2 = []
        for sample in self.samples:

            img = f'JPEGImages/{str(sample)}.jpg'
            label = f'{self.anno_type}/{str(sample)}.png'

            label_ = np.array(Image.open(os.path.join(self.dataset_root_dir, label)), dtype=np.int8)
            H, W = label_.shape

            sample = dict()
            sample['images'] = img
            sample['labels'] = label

            if H < W:
                samples_list_1.append(sample)
            else:
                samples_list_2.append(sample)

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

        images = self.samples_list[idx]['images']
        labels = self.samples_list[idx]["labels"]
        sample_name = images.split('/')[1].split('.')[0]
        # images
        images = np.array(Image.open(os.path.join(self.dataset_root_dir, images)).convert('RGB'))
        labels = np.array(Image.open(os.path.join(self.dataset_root_dir, labels)), dtype=np.int8)
        labels[labels == 255] = -1

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
def slide_window_cls_feat_clustering(cfg, feat_save_dir, dataset, device, _log):
    N_things = cfg.model.decoder.n_things
    N_stuff = cfg.model.decoder.n_stuff
    h = w = cfg.dataset.resize // cfg.model.encoder.patch_size
    cluster_centroids_fg_save_path = os.path.join(feat_save_dir, f'cluster_centroids_slide_window_fg_{N_things}_{h}x{w}')
    if not os.path.exists(cluster_centroids_fg_save_path):
        samples_list = dataset.samples_list
        cls_feature_fg_all = []
        for i, sample in enumerate(samples_list):
            file_name = sample['images'].split('/')[1].split('.')[0]
            cls_feat_fg_save_path = os.path.join(feat_save_dir, f"{file_name}_cls_feat_fg_{N_things}_{h}x{w}")
            if os.path.exists(cls_feat_fg_save_path):
                dictfile = open(cls_feat_fg_save_path, 'rb')
                cls_feat_k = pickle.load(dictfile)
                cls_feature_fg_all.append(torch.from_numpy(cls_feat_k))
                _log.info(f"id {i+1}/{len(samples_list)} fg feature loaded from {cls_feat_fg_save_path}")
            else:
                _log.info(f"{i+1}/{len(samples_list)} fg feature do not found at {cls_feat_fg_save_path}")
        cls_feature_fg_all = torch.cat(cls_feature_fg_all).to(device)
        cluster_centroids_fg, labels_fg = train_kmeans(cls_feature_fg_all, N_things, metric='l2')

        dictfile = open(cluster_centroids_fg_save_path, 'wb')
        pickle.dump(cluster_centroids_fg.cpu().numpy(), dictfile)
        dictfile.close()
        _log.info(f"clustering done, save cluster_centroids to {cluster_centroids_fg_save_path}")
    else:
        cluster_centroids_fg = pickle.load(open(cluster_centroids_fg_save_path, 'rb'))
        cluster_centroids_fg = torch.from_numpy(cluster_centroids_fg)
        _log.info(f"load cluster_centroids_fg from {cluster_centroids_fg_save_path}")

    return cluster_centroids_fg


@ex.capture
def generate_pseudo_label_pascalvoc(cfg, model, data_loader, cluster_centroids_fg, pseudo_label_save_dir, device, _log):
    batch_time = AverageMeter()
    tic = time.time()
    model.train()
    N_things = cfg.model.decoder.n_things
    assert cluster_centroids_fg.shape[0] == N_things
    for index, sample in enumerate(data_loader):
        images = sample['images'].float().to(device)   # image, normalized
        sample_names = sample['meta']['sample_name']

        pseudo_labels_fg = model(images, cluster_centroids=cluster_centroids_fg[None].expand(images.shape[0], -1, -1).to(device))
        N_cls, h, w = pseudo_labels_fg.shape[1:]
        assert pseudo_labels_fg.shape[1:] == (N_things, h, w)
        assert h == w
        for sample_name, pseudo_label_fg in zip(sample_names, pseudo_labels_fg):

            pseudo_label_fg_save_path = os.path.join(pseudo_label_save_dir, sample_name+f"_fg_{N_things}_{h}x{w}")
            pseudo_label_fg = pseudo_label_fg.cpu().numpy()
            dictfile = open(pseudo_label_fg_save_path, 'wb')
            pickle.dump(pseudo_label_fg, dictfile)
            dictfile.close()

        # update time
        batch_time.update(time.time() - tic)
        tic = time.time()

        _log.info(f"[{index + 1:4d}/{len(data_loader):4d}]\t"
                  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                  f"N_things: {N_things}\tsize:{h}x{w}")

    _log.info(f"Done")


@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_bank_dir = os.path.join(cfg.dataset.root_dir_cls_feat, f'cls_feat_pascalvoc')
    pseudo_label_save_dir = os.path.join(cfg.dataset.root_dir_pseudo_label, f'pseudo_label_pascalvoc')

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
    train_dataset_pascalvoc = load_train_dataset_pascalvoc(cfg, split=cfg.dataset.split, sample_range=sample_range)

    cluster_centroids_fg = slide_window_cls_feat_clustering(cfg, feature_bank_dir, train_dataset_pascalvoc, device)
    sampler = None
    train_loader = DataLoader(train_dataset_pascalvoc, batch_size=cfg.dataset.train_batch_size,
                              shuffle=False,
                              num_workers=cfg.dataset.num_workers,
                              pin_memory=True, drop_last=False, sampler=sampler)
    generate_pseudo_label_pascalvoc(cfg, model, train_loader, cluster_centroids_fg, pseudo_label_save_dir, device)
