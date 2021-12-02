'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import shutil
import numpy as np
import os
import cv2
from collections import OrderedDict
import torch.distributed as dist
import hostlist
from pathlib import Path

import torch



def init_process(backend="nccl"):
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"Starting process with rank {dist_rank}...", flush=True)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["MASTER_PORT"] = str(12345 + int(gpu_ids[0]))
    elif "SLURM_STEPS_GPUS" in os.environ:
        gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
        os.environ["MASTER_PORT"] = str(12345 + int(min(gpu_ids)))
    else:
        os.environ["MASTER_PORT"] = str(12345) + np.random.randint(0, 100)

    if "SLURM_JOB_NODELIST" in os.environ:
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group(
        backend,
        rank=dist_rank,
        world_size=world_size,
    )
    print(f"Process {dist_rank} is connected, world_size is {world_size}", flush=True)
    dist.barrier()

    # silence_print(dist_rank == 0)
    # if dist_rank == 0:
    #     print(f"All processes are connected.", flush=True)


def get_vit_vistr_params_groups(model, cfg=None):
    decoder_regularized = []
    decoder_not_regularized = []
    encoder_regularized = []
    encoder_not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            if 'encoder' in name:
                encoder_not_regularized.append(param)
            else:
                decoder_not_regularized.append(param)
        else:
            if 'encoder' in name:
                encoder_regularized.append(param)
            else:
                decoder_regularized.append(param)
    return [
        {'params': encoder_regularized, 'lr': cfg.lr*0.1},
        {'params': encoder_not_regularized, 'weight_decay': 0., 'lr': cfg.lr*0.1},
        {'params': decoder_regularized},
        {'params': decoder_not_regularized, 'weight_decay': 0.},
    ]


def get_params_groups(model, cfg=None):
    decoder_regularized = []
    decoder_not_regularized = []
    encoder_regularized = []
    encoder_not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            if 'encoder' in name and not cfg.model.encoder.fix:
                encoder_not_regularized.append(param)
            elif 'decoder' in name:
                decoder_not_regularized.append(param)
            else:
                pass
        else:
            if 'encoder' in name and not cfg.model.encoder.fix:
                encoder_regularized.append(param)
            elif 'decoder' in name:
                decoder_regularized.append(param)
            else:
                pass
    return [
        {'params': encoder_regularized, 'lr': cfg.lr*cfg.model.encoder.lr_scale},
        {'params': encoder_not_regularized, 'weight_decay': 0., 'lr': cfg.lr*cfg.model.encoder.lr_scale},
        {'params': decoder_regularized},
        {'params': decoder_not_regularized, 'weight_decay': 0.},
    ]

def sync_model(sync_dir, model):
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    sync_path = Path(sync_dir).resolve() / "sync_model.pkl"
    if dist_rank == 0 and world_size > 1:
        torch.save(model.state_dict(), sync_path)
    dist.barrier()
    if dist_rank > 0:
        model.load_state_dict(torch.load(sync_path))
    dist.barrier()
    if dist_rank == 0 and world_size > 1:
        sync_path.unlink()
    return model



class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_network_checkpoint(ckpt_dir, encoder, decoder=None, is_best=False):

    dict_network = encoder.state_dict()
    new_state_dict = OrderedDict()
    for k, v in dict_network.items():
        name = k[7:] if 'module' in k else k  # remove `module.`
        new_state_dict[name] = v.clone().detach().to('cpu')
    torch.save(dict_network, os.path.join(ckpt_dir, 'encoder_latest.pt'))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, 'encoder_latest.pt'),
                        os.path.join(ckpt_dir, 'encoder_best.pt'))

    if decoder is not None:
        dict_network = decoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in dict_network.items():
            name = k[7:] if 'module' in k else k  # remove `module.`
            new_state_dict[name] = v.clone().detach().to('cpu')
        torch.save(dict_network, os.path.join(ckpt_dir, 'decoder_latest.pt'))

        if is_best:
            shutil.copyfile(os.path.join(ckpt_dir, 'decoder_latest.pt'),
                            os.path.join(ckpt_dir, 'decoder_best.pt'))

def load_network_checkpoint(ckpt_dir, encoder=None, decoder=None, is_best=True, strict=True):
    if encoder is not None:
        if is_best:
            encoder_state_dict = torch.load(os.path.join(ckpt_dir, f'encoder_best.pt'), map_location=torch.device('cpu'))
        else:
            encoder_state_dict = torch.load(os.path.join(ckpt_dir, f'encoder_latest.pt'), map_location=torch.device('cpu'))

        new_encoder_state_dict = OrderedDict()
        for k, v in encoder_state_dict.items():
            name = k[7:] if 'module' in k else k  # remove `module.`
            new_encoder_state_dict[name] = v
        encoder.load_state_dict(new_encoder_state_dict, strict=strict)

    if decoder is not None:
        if is_best:
            decoder_state_dict = torch.load(os.path.join(ckpt_dir, f'decoder_best.pt'), map_location=torch.device('cpu'))
        else:
            decoder_state_dict = torch.load(os.path.join(ckpt_dir, f'decoder_latest.pt'), map_location=torch.device('cpu'))
        new_decoder_state_dict = OrderedDict()
        for k, v in decoder_state_dict.items():
            name = k[7:] if 'module' in k else k  # remove `module.`
            new_decoder_state_dict[name] = v
        decoder.load_state_dict(new_decoder_state_dict, strict=strict)





def generate_video_from_images(save_dir: str, seq_name: str, img_array, fps=15, video_format='mp4'):
    '''
    save_path: path to the dest dir
    img_array: T x 3 x H x W, numpy array, RGB, int(0-255)
    '''
    os.makedirs(save_dir, exist_ok=True)
    FOURCC = {
        "mp4": cv2.VideoWriter_fourcc(*"MP4V"),
        "avi": cv2.VideoWriter_fourcc(*"XVID"),
    }

    size = img_array[0].shape[-3:-1][::-1]
    path = os.path.join(save_dir, f"{seq_name}." + video_format)
    print(f"Generating video {size} to {path}")

    dest = cv2.VideoWriter(
        path,
        FOURCC[video_format],
        fps,
        size,
    )

    for i in range(len(img_array)):
        dest.write(img_array[i])
    dest.release()
    print("Done")
