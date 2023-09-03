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
import copy
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.optimize import linear_sum_assignment as linear_assignment

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as pth_transforms


from models.segmenter.segmenter_things import Segmenter as Model

from dataloaders.PrefetchLoader import PrefetchLoader
from dataloaders import transforms_uss_thingstuff
from utils.misc import AverageMeter, init_process, sync_model, get_params_groups, save_network_checkpoint
from utils.metric import scores, get_result_metrics


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


@ex.capture
def load_train_dataset_pascalvoc(cfg, split='train', pseudo_label_save_dir=None, num_samples=None, _log=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.ToTensor(),
        transforms_uss_thingstuff.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])


    _log.info(f"load pseudo_label from {pseudo_label_save_dir}")
    dataset = PascalVOC(transform=train_transform,
                        split=split,
                        dataset_root_dir=cfg.dataset.root_dir_pascalvoc,
                        pseudo_label_save_dir=pseudo_label_save_dir,
                        pseudo_label_size=cfg.dataset.pseudo_label_size,
                        num_things=cfg.model.decoder.n_things,
                        num_samples=num_samples,
                        anno_type='instance&object',
                        orientation=0)

    return dataset


def load_val_dataset_pascalvoc(cfg, split='val', pseudo_label_save_dir=None, num_samples=None):

    train_transform = pth_transforms.Compose([
        transforms_uss_thingstuff.NormInput(),
        transforms_uss_thingstuff.ToTensor(),
        transforms_uss_thingstuff.ResizeTensor(size=(cfg.dataset.resize, cfg.dataset.resize), img_only=False),
    ])

    dataset = PascalVOC(transform=train_transform,
                        split=split,
                        dataset_root_dir=cfg.dataset.root_dir_pascalvoc,
                        pseudo_label_save_dir=pseudo_label_save_dir,
                        pseudo_label_size=cfg.dataset.pseudo_label_size,
                        num_things=cfg.model.decoder.n_things,
                        num_samples=num_samples,
                        anno_type='instance&object',
                        orientation=0)
    return dataset

class PascalVOC(Dataset):
    def __init__(self,
                 split=None,
                 dataset_root_dir=None,
                 pseudo_label_save_dir=None,
                 pseudo_label_size=40,
                 num_things=80,
                 transform=None,
                 num_samples=None,
                 orientation=0,
                 anno_type='instances',
                 ):
        assert split in ['train', 'val']
        self.split = split
        self.dataset_root_dir = dataset_root_dir
        self.pseudo_label_save_dir = pseudo_label_save_dir
        self.pseudo_label_size = pseudo_label_size
        self.transform = transform
        self.num_samples = num_samples
        self.anno_type = 'SegmentationClass'
        self.num_things = num_things

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
        self.samples_list = samples_list


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        images = self.samples_list[idx]['images']
        labels = self.samples_list[idx]["labels"]
        sample_name = images.split('/')[1].split('.')[0]

        if self.pseudo_label_save_dir is not None:
            pseudo_label_things_save_path = os.path.join(self.pseudo_label_save_dir, sample_name +f'_fg_{self.num_things}_{self.pseudo_label_size}x{self.pseudo_label_size}')
            assert os.path.exists(pseudo_label_things_save_path)

        # images
        images = np.array(Image.open(os.path.join(self.dataset_root_dir, images)).convert('RGB'))
        labels = np.array(Image.open(os.path.join(self.dataset_root_dir, labels)), dtype=np.int8)
        labels[labels == 255] = -1

        sample_ = dict()
        sample_['img'] = images
        sample_['label_cat'] = labels
        sample_['meta'] = {'sample_name': sample_name}

        if self.pseudo_label_save_dir is not None:
            pseudo_label_things_save_path = os.path.join(self.pseudo_label_save_dir, sample_name +f'_fg_{self.num_things}_{self.pseudo_label_size}x{self.pseudo_label_size}')
            sample_['pseudo_label_things'] = pickle.load(open(pseudo_label_things_save_path, 'rb'))

        if self.transform is not None:
            sample_ = self.transform(sample_)

        sample = dict()
        sample['images'] = sample_['img']
        sample['label_cat'] = sample_['label_cat']
        sample['meta'] = sample_['meta']
        if self.pseudo_label_save_dir is not None:
            sample['pseudo_label_things'] = sample_['pseudo_label_things']

        return sample


@ex.capture
def train_pascalvoc(cfg, model, model_init_weights, optimizer, data_loader, history, device, epoch, epoch_iter, _log):
    batch_time = AverageMeter()
    losses_all = AverageMeter()
    losses_cat = AverageMeter()
    losses_uncertainty = AverageMeter()
    losses_cls_emb = AverageMeter()
    tic = time.time()
    model.train()
    epoch_step = 10
    bootstrapping_start_epoch = cfg.model.bootstrapping_start_epoch
    intervals = torch.Tensor([int(i) for i in str(cfg.model.teacher_update_interval).split(',')])
    teacher_update_interval = intervals[0] if len(intervals) == 1 else intervals[min(epoch // epoch_step, len(intervals) - 1)]
    if not epoch < bootstrapping_start_epoch and epoch % teacher_update_interval.item() == 0:
        _log.info(f"Epoch {epoch:2d}, update teacher and reboot student, reset epoch_iter")
        epoch_iter = 0
        if hasattr(model, 'module'):
            model.module.encoder_teacher.load_state_dict(copy.deepcopy(model.module.encoder.state_dict()))
            model.module.decoder_teacher.load_state_dict(copy.deepcopy(model.module.decoder.state_dict()))
            model.module.encoder.load_state_dict(copy.deepcopy(model_init_weights['encoder_init_weights']))
            model.module.decoder.load_state_dict(copy.deepcopy(model_init_weights['decoder_init_weights']))
        else:
            model.encoder_teacher.load_state_dict(copy.deepcopy(model.encoder.state_dict()))
            model.decoder_teacher.load_state_dict(copy.deepcopy(model.decoder.state_dict()))
            model.encoder.load_state_dict(copy.deepcopy(model_init_weights['encoder_init_weights']))
            model.decoder.load_state_dict(copy.deepcopy(model_init_weights['decoder_init_weights']))
    else:
        epoch_iter = epoch_iter + 1

    for index, sample in enumerate(data_loader):
        images = sample['images'].float().to(device, non_blocking=True)   # image, normalized
        label_cat = sample['label_cat'].float().to(device, non_blocking=True)   # label
        pseudo_label_things = sample['pseudo_label_things'].float().to(device, non_blocking=True)   # things pseudo label

        assert cfg.model.decoder.n_things == pseudo_label_things.shape[1]
        optimizer.zero_grad()
        losses = model(images, return_loss=True, label=label_cat, pseudo_labels=pseudo_label_things,
                       bootstrapping=True if not epoch < bootstrapping_start_epoch else False,
                       augment=True, epoch=epoch_iter)
        loss_cat = losses['loss_cat'].mean()
        loss_uncertainty = losses['loss_uncertainty'].mean()
        loss_cls_emb = losses['loss_cls_emb'].mean()

        w = [float(w) for w in str(cfg.model.loss.weights).split(',')]
        w = torch.Tensor(w).to(device)
        assert len(w) == 3
        loss = w[0] * loss_cat + w[1] * loss_uncertainty + w[2] * loss_cls_emb
        if loss > 0:
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()
            losses_all.update(loss.detach().item())
            losses_cat.update(loss_cat.detach().item())
            losses_uncertainty.update(loss_uncertainty.detach().item())
            losses_cls_emb.update(loss_cls_emb.detach().item())

        # update time
        batch_time.update(time.time() - tic)
        tic = time.time()

        _log.info(f"train epoch: [{epoch}][{index + 1:4d}/{len(data_loader):4d}]\t"
                  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                  f"Loss(cat/unc/emb/all): {losses_cat.val:.4f}/{losses_uncertainty.val:.4f}/{losses_cls_emb.val:.4f}/{losses_all.val:.4f} "
                  f"({losses_cat.avg:.4f}/{losses_uncertainty.avg:.4f}/{losses_cls_emb.avg:.4f}/{losses_all.avg:.4f})")

    _log.info(f"* train epoch: [{epoch}]\t"
              f"loss(cat/unc/emb/all): {losses_cat.avg:.4f}/{losses_uncertainty.avg:.4f}/{losses_cls_emb.avg:.4f}/{losses_all.avg:.4f}")
    history['train']['loss'].append(losses_all.avg)

    return epoch_iter


@ex.capture
def eval_pascalvoc(cfg, model, data_loader, history, device, epoch, exp_ckpt_dir, _log):
    batch_time = AverageMeter()
    tic = time.time()
    model.eval()

    N_things = cfg.model.decoder.n_things
    N_stuff = 1
    N_cls = N_things + N_stuff
    histogram = np.zeros((N_cls, N_cls))
    for index, sample in enumerate(data_loader):
        images = sample['images'].float().to(device, non_blocking=True)   # image, normalized
        label_cat = sample['label_cat'].int().to(device, non_blocking=True)   # label
        N, C, H, W = images.shape

        with torch.no_grad():

            probs = model(images)

            probs = F.interpolate(probs, size=(H, W), mode='bilinear', align_corners=False)
            preds = probs.topk(1, dim=1)[1].view(N, -1).cpu().numpy()
            label_cat_ = label_cat.view(N, -1).cpu().numpy()
            histogram += scores(label_cat_, preds, N_cls)

        # update time
        batch_time.update(time.time() - tic)
        tic = time.time()

        _log.info(f"eval epoch: [{epoch}][{index + 1:4d}/{len(data_loader):4d}]\t"
                  f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f})")

    # Hungarian Matching.
    m = linear_assignment(histogram.max() - histogram)

    new_hist = np.zeros((N_cls, N_cls))
    for idx in range(N_cls):
        new_hist[m[1][idx]] = histogram[idx]

    res = get_result_metrics(new_hist)
    _log.info(f"ACC  - All: {res['overall_precision (pixel accuracy)']:.4f}")
    _log.info(f"mIOU - All: {res['mean_iou']:.4f}")

    _log.info(f"* eval epoch: [{epoch}]\tACC: {res['overall_precision (pixel accuracy)']:.4f}\t"
              f"mIoU: {res['mean_iou']:.4f}")
    history['val']['metric'].append(res['mean_iou'] + res['overall_precision (pixel accuracy)'])

    if cfg.eval_only:
        generate_and_save_vis(model, data_loader, device, m, exp_ckpt_dir)


@ex.capture
def generate_and_save_vis(model, data_loader, device, m, save_root_dir, _log):

    mean = torch.Tensor((0.485, 0.456, 0.406))[:, None, None].to(device)
    std = torch.Tensor((0.229, 0.224, 0.225))[:, None, None].to(device)

    map = np.vectorize(lambda x: {i: id for i, id in enumerate(m[0][np.argsort(m[1])])}[x])
    from utils.colormap import colormap
    import cv2
    save_dir = os.path.join(save_root_dir, 'visualization')
    os.makedirs(save_dir, exist_ok=True)
    for index, sample in enumerate(data_loader):
        images = sample['images'].float().to(device, non_blocking=True)   # image, normalized
        label_cat = sample['label_cat'].int().to(device, non_blocking=True)   # label
        N, C, H, W = images.shape
        with torch.no_grad():
            probs = model(images)
            probs = F.interpolate(probs, size=(H, W), mode='bilinear', align_corners=False)
            masks_ = probs.max(dim=1)[1].detach().cpu()

            images_ = (((images * std) + mean) * 255).int()
            for mask, image, label, name in zip(masks_, images_, label_cat, sample['meta']['sample_name']):
                image_ = image.permute(1, 2, 0).cpu().numpy()
                mask_ = colormap[map(mask.cpu())]
                label_ = colormap[label.cpu()]
                vis = np.concatenate([image_, mask_, label_], 1)
                cv2.imwrite(f"{save_dir}/{name}.png", vis[:, :, ::-1])

        _log.info(f"vis batch [{index + 1:4d}/{len(data_loader):4d}]")
    _log.info(f"visualization saved into {save_dir}")




@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_ckpt_dir = os.path.join(_run.meta_info['options']['--file_storage'], str(_run._id)) if _run._id else os.path.join('train', 'public')
    pseudo_label_train_save_dir = os.path.join(cfg.dataset.root_dir_pseudo_label, f'pseudo_label_pascalvoc')
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    # Network Builders
    model = Model(cfg.model)
    if cfg.eval_only:
        pretrain_model = cfg.model.decoder.pretrained_weight
        assert os.path.exists(pretrain_model)
        decoder_state_dict = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.decoder.load_state_dict(decoder_state_dict, strict=True)
        print(f"load pretrained model from {pretrain_model}")
    else:
        assert os.path.exists(pseudo_label_train_save_dir)

    model_init_weights = {'encoder_init_weights': copy.deepcopy(model.encoder.state_dict()),
                          'decoder_init_weights': copy.deepcopy(model.decoder.state_dict())}
    use_ddp = cfg.model.use_ddp == 1
    if torch.cuda.is_available():
        model.cuda()
        if torch.cuda.device_count() > 1:
            _log.info(f"using {torch.cuda.device_count()} gpus")
            if use_ddp:
                init_process()
                model = DDP(model, find_unused_parameters=True)
                sync_model('sync_dir', model)
            else:
                model = DP(model)

    paras = get_params_groups(model, cfg)
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(paras, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError

    train_dataset_pascalvoc = load_train_dataset_pascalvoc(cfg, split=cfg.dataset.split, pseudo_label_save_dir=pseudo_label_train_save_dir)
    if cfg.dataset.repeat > 0:
        train_dataset_pascalvoc = ConcatDataset([train_dataset_pascalvoc for _ in range(cfg.dataset.repeat)])
    val_dataset_pascalvoc = load_val_dataset_pascalvoc(cfg, split='val', pseudo_label_save_dir=None)
    val_loader_pascalvoc = DataLoader(val_dataset_pascalvoc, batch_size=cfg.dataset.val_batch_size,
                                 shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    if torch.cuda.is_available():
        val_loader_pascalvoc = PrefetchLoader(val_loader_pascalvoc)
    # save losses per epoch
    history = {'train': {'loss': [], 'metric_pred': 0, 'metric_pseudo_label': 0, },
               'val': {'metric': [], 'best_metric': 0}}

    if cfg.eval_only:
        with torch.no_grad():
            eval_pascalvoc(cfg, model, val_loader_pascalvoc, history, device, 0, exp_ckpt_dir)
    else:
        epoch_iter = 0
        for epoch in range(cfg.num_epochs):

            sampler = DistributedSampler(train_dataset_pascalvoc, shuffle=True) if torch.cuda.device_count() > 1 and use_ddp else None
            train_loader = DataLoader(train_dataset_pascalvoc, batch_size=cfg.dataset.train_batch_size,
                                      shuffle=False if sampler else True,
                                      num_workers=cfg.dataset.num_workers,
                                      prefetch_factor=4,
                                      persistent_workers=True,
                                      pin_memory=True, drop_last=True, sampler=sampler)
            if torch.cuda.is_available():
                train_loader = PrefetchLoader(train_loader)
            epoch_iter = train_pascalvoc(cfg, model, model_init_weights, optimizer, train_loader, history, device, epoch, epoch_iter, )

            if (epoch + 1) % cfg.eval_interval == 0:
                with torch.no_grad():
                    eval_pascalvoc(cfg, model, val_loader_pascalvoc, history, device, epoch, None)
                    if history['val']['metric'][-1] > history['val']['best_metric']:
                        history['val']['best_metric'] = history['val']['metric'][-1]
                        save_network_checkpoint(exp_ckpt_dir, model.module.encoder, model.module.decoder, is_best=True)
                    else:
                        save_network_checkpoint(exp_ckpt_dir, model.module.encoder, model.module.decoder, is_best=False)