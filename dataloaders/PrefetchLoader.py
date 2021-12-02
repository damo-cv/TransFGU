'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
'''
import torch

class PrefetchLoader():

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_data in self.loader:
            with torch.cuda.stream(stream):
                next_data['images'] = next_data['images'].cuda(non_blocking=True)
                next_data['label_cat'] = next_data['label_cat'].cuda(non_blocking=True)
                if 'pseudo_label_things' in next_data.keys():
                    next_data['pseudo_label_things'] = next_data['pseudo_label_things'].cuda(non_blocking=True)
                if 'pseudo_label_stuff' in next_data.keys():
                    next_data['pseudo_label_stuff'] = next_data['pseudo_label_stuff'].cuda(non_blocking=True)
                if 'pseudo_label_gt' in next_data.keys():
                    next_data['pseudo_label_gt'] = next_data['pseudo_label_gt'].cuda(non_blocking=True)

            if not first:
                yield data
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data

        yield data

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
