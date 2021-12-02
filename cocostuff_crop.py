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
from dataloaders.coco_id_idx_map import coco_stuff_id_idx_map
from utils.misc import AverageMeter

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


def load_val_dataset_coco(cfg, split='val', sample_range=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.Resize(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
        transforms_uss_thingstuff.NormInput(),
        transforms_uss_thingstuff.ToTensor(),
    ])

    dataset = MSCOCO17(transform=train_transform,
                         split=split,
                         dataset_root_dir=cfg.dataset.root_dir_mscoco,
                         sample_range=sample_range,
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
                 ):
        assert split in ['train', 'val']
        self.split = 'train2017' if split == 'train' else 'val2017'
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.num_samples = num_samples

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

        annIds = self.coco.getAnnIds(imgIds=id)
        instance = self.coco.loadAnns(annIds)

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
def crop_image_slide_window_fgbg(images, fg_masks, anchors_fg, anchors_bg, h, w, th_fg=0.5, th_bg=0.8, _log=None):
    '''

    '''
    N, _, H, W = images.shape
    bg_masks = 1 - fg_masks
    image_fg_crops_list = []
    image_bg_crops_list = []
    N_anchors_fg = anchors_fg.shape[0]
    N_anchors_bg = anchors_bg.shape[0]
    anchors_fg = F.interpolate(anchors_fg.view(N_anchors_fg, h, w)[None], (H, W))[0]
    anchors_bg = F.interpolate(anchors_bg.view(N_anchors_bg, h, w)[None], (H, W))[0]
    fg_masks = F.interpolate(fg_masks.view(N, 1, h, w).float(), (H, W))[:, 0]
    bg_masks = F.interpolate(bg_masks.view(N, 1, h, w).float(), (H, W))[:, 0]
    for image, fg_mask, bg_mask in zip(images, fg_masks, bg_masks):
        image_fg_crops = []
        image_bg_crops = []
        for i, anchor in enumerate(anchors_fg):
            xy = anchor.nonzero()
            assert len(xy) > 0
            x0, x1, y0, y1 = xy[:, 1].min(), xy[:, 1].max(), xy[:, 0].min(), xy[:, 0].max()
            if (fg_mask[y0:y1, x0:x1]).sum() / ((y1 - y0) * (x1 - x0)) > th_fg:
                image_fg_crops.append(F.interpolate(image[:, y0:y1, x0:x1][None], (H // 2, W // 2), mode='bilinear'))
        for i, anchor in enumerate(anchors_bg):
            xy = anchor.nonzero()
            assert len(xy) > 0
            x0, x1, y0, y1 = xy[:, 1].min(), xy[:, 1].max(), xy[:, 0].min(), xy[:, 0].max()
            if (bg_mask[y0:y1, x0:x1]).sum() / ((y1 - y0) * (x1 - x0)) > th_bg:
                image_bg_crops.append(F.interpolate(image[:, y0:y1, x0:x1][None], (H // 2, W // 2), mode='bilinear'))
        if len(image_fg_crops) > 0:
            image_fg_crops = torch.cat(image_fg_crops)
            image_fg_crops_list.append(image_fg_crops)
        if len(image_bg_crops) > 0:
            image_bg_crops = torch.cat(image_bg_crops)
            image_bg_crops_list.append(image_bg_crops)
    return image_fg_crops_list, image_bg_crops_list


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
def crop_and_extract_features_cocostuff_fgbg_slide_window(cfg, model, model_val, data_loader, device, feat_save_dir, _log):
    batch_time = AverageMeter()
    tic = time.time()
    model.eval()

    batch_size_crops = cfg.dataset.train_batch_size
    anchors_fg = generate_box_anchors_by_scale(cfg.dataset.resize // cfg.model.encoder.patch_size,
                                            cfg.dataset.resize // cfg.model.encoder.patch_size,
                                            scales=(0.5, 0.4, 0.3, 0.2))
    anchors_bg = generate_box_anchors_by_scale(cfg.dataset.resize // cfg.model.encoder.patch_size,
                                            cfg.dataset.resize // cfg.model.encoder.patch_size,
                                            scales=(0.5, 0.4, 0.3, 0.2))
    anchors_fg = anchors_fg.to(device)
    anchors_bg = anchors_bg.to(device)
    for index, sample in enumerate(data_loader):

        images = sample['images'].float().to(device)   # image, normalized
        meta = sample['meta']
        sample_names = sample['meta']['sample_name']
        N, _, H, W = images.shape
        N_cls = cfg.model.decoder.n_things + cfg.model.decoder.n_stuff
        h, w = H // cfg.model.encoder.patch_size, W // cfg.model.encoder.patch_size
        embed_dim = model_val.embed_dim
        for image, sample_name in zip(images, sample_names):
            _, attns = model_val.get_intermediate_layers_feat_attn(image[None], 1)
            attn = attns[-1].mean(1)[:, 0, 1:].view(N, h, w)
            attn_bin = (attn > attn.mean()).int()
            image_fg_crops, image_bg_crops = crop_image_slide_window_fgbg(image[None], attn_bin, anchors_fg, anchors_bg, h, w)
            if len(image_fg_crops) > 0:
                image_crops = torch.cat(image_fg_crops)
                N_crops, _, H_crop, W_crop = image_crops.shape
                h_crop, w_crop = H_crop // cfg.model.encoder.patch_size, W_crop // cfg.model.encoder.patch_size
                sample_name_fg = sample_name+f"_cls_feat_fg_{h_crop}x{w_crop}"
                extract_features_and_save(model, model_val, image_crops, batch_size_crops, embed_dim, device, H, W,
                                          feat_save_dir, sample_name_fg)
            if len(image_bg_crops) > 0:
                image_crops = torch.cat(image_bg_crops)
                N_crops, _, H_crop, W_crop = image_crops.shape
                h_crop, w_crop = H_crop // cfg.model.encoder.patch_size, W_crop // cfg.model.encoder.patch_size
                sample_name_bg = sample_name+f"_cls_feat_bg_{h_crop}x{w_crop}"
                extract_features_and_save(model, model_val, image_crops, batch_size_crops, embed_dim, device, H, W,
                                          feat_save_dir, sample_name_bg)

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

    feature_bank_dir = os.path.join(cfg.dataset.root_dir_cls_feat, f'cls_feat_cocostuff')
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
        dataset_coco = load_val_dataset_coco(cfg, split=cfg.dataset.split, sample_range=sample_range)
        loader_coco = DataLoader(dataset_coco, batch_size=1, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

        model_val = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            crop_and_extract_features_cocostuff_fgbg_slide_window(cfg, model, model_val, loader_coco, device, feature_bank_dir, )