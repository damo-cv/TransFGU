# TransFGU: A Top-down Approach to Fine-Grained Unsupervised Semantic Segmentation
[Zhaoyun Yin](https://github.com/zyy-cn),
[Pichao Wang](https://sites.google.com/site/pichaossites/),
[Fan Wang](https://scholar.google.com/citations?user=WCRGTHsAAAAJ&hl=en),
Xianzhe Xu,
[Hanling Zhang](http://design.hnu.edu.cn/info/1023/5767.htm),
[Hao Li](https://scholar.google.com/citations?user=pHN-QIwAAAAJ&hl=zh-CN),
[Rong Jin](https://scholar.google.com/citations?user=CS5uNscAAAAJ&hl=zh-CN)

[[Preprint](https://arxiv.org/abs/2112.01515)]

## Getting Started

Create the environment

```bash
# create conda env
conda create -n TransFGU python=3.8
# activate conda env
conda activate TransFGU
# install pytorch
conda install pytorch=1.8 torchvision cudatoolkit=10.1
# install other dependencies
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html
pip install -r requirements.txt
```

## Dataset Preparation

- MS-COCO Dataset: Download the [trainset](http://images.cocodataset.org/zips/train2017.zip), [validset](http://images.cocodataset.org/zips/val2017.zip), [annotations](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) and the [json files](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), place the extracted files into `root/data/MSCOCO`.
- PascalVOC Dataset: Download [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), place the extracted files into `root/data/PascalVOC`.
- Cityscapes Dataset: Download [leftImg8bit_trainvaltest.zip](http://images.cocodataset.org/zips/train2017.zip) and [gtFine_trainvaltest.zip](http://images.cocodataset.org/zips/val2017.zip), place the extracted files into `root/data/Cityscapes`.
- LIP Dataset: Download [TrainVal_images.zip](https://drive.google.com/file/d/0BzvH3bSnp3E9cVl3b3pKdmFlclE/view?usp=sharing&resourcekey=0-0crLyOF_Tj-hPDLS89DtqA) and [TrainVal_parsing_annotations.zip](https://drive.google.com/file/d/15tifhBogDs_oBUKaUf362vzZTlIdzktv/view?usp=sharing), place the extracted files into `root/data/LIP`.

the structure of dataset folders should be as follow:
~~~
data/
    │── MSCOCO/
    │     ├── images/
    │     │     ├── train2017/
    │     │     └── val2017/
    │     └── annotations/
    │           ├── train2017/
    │           ├── val2017/
    │           ├── instances_train2017.json
    │           └── instances_val2017.json
    │── Cityscapes/
    │     ├── leftImg8bit/
    │     │     ├── train/
    │     │     │       ├── aachen
    │     │     │       └── ...
    │     │     └──── val/
    │     │             ├── frankfurt
    │     │             └── ...
    │     └── gtFine/
    │           ├── train/
    │           │       ├── aachen
    │           │       └── ...
    │           └──── val/
    │                   ├── frankfurt
    │                   └── ...
    │── PascalVOC/
    │     ├── JPEGImages/
    │     ├── SegmentationClass/
    │     └── ImageSets/
    │           └── Segmentation/
    │                   ├── train.txt
    │                   └── val.txt
    └── LIP/
          ├── train_images/
          ├── train_segmentations/
          ├── val_images/
          ├── val_segmentations/
          ├── train_id.txt
          └── val_id.txt
~~~


## Model download
- please download the pretrained [dino model (deit small 8x8)](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth), then place it into `root/weight/dino/` 
- download trained model from [Google Drive](https://drive.google.com/drive/folders/1vHKLrAE51mLTK-5DpzByQ_g1RAjmONyi?usp=sharing) or [Baidu Netdisk (code:1118)](https://pan.baidu.com/s/1N7GSzcMOi9C3mgpUsIa4oA), then place them into `root/weight/trained/` 

<table>
  <tr>
    <th>Name</th>
    <th>mIoU</th>
    <th>Pixel Accuracy</th>
    <th>Model</th>
  </tr>
  <tr>
    <td>COCOStuff-27</td>
    <td>16.19</td>
    <td>44.52</td>
    <td><a href="https://drive.google.com/file/d/1cEQj1YqxbxrechgbWcRsbLD1tGVSKOkx/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>COCOStuff-171</td>
    <td>11.93</td>
    <td>34.32</td>
    <td><a href="https://drive.google.com/file/d/1NCGDHDS1gSIiI02dtbolcNY63ohOvMLM/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>COCO-80</td>
    <td>12.69</td>
    <td>64.31</td>
    <td><a href="https://drive.google.com/file/d/1v1ogrw68DGCSU72CGqMq7h6aYYM13NUU/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>Cityscapes</td>
    <td>16.83</td>
    <td>77.92</td>
    <td><a href="https://drive.google.com/file/d/10Nh3uONXZ5DspEzbFMoOIKSVe6w2i6ya/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>Pascal-VOC</td>
    <td>37.15</td>
    <td>83.59</td>
    <td><a href="https://drive.google.com/file/d/1qJDIa-4lTP6-HxArJhk-DLjQOnCYq5p2/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>LIP-5</td>
    <td>25.16</td>
    <td>65.76</td>
    <td><a href="https://drive.google.com/file/d/1yqsg2CX6KxnDVlnD1TgoWQqEGcfG8O3x/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>LIP-16</td>
    <td>15.49</td>
    <td>60.08</td>
    <td><a href="https://drive.google.com/file/d/1AUbDQ0T1bhPE0GtIGvRe-GkTv0zymw45/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>LIP-19</td>
    <td>12.24</td>
    <td>42.52</td>
    <td><a href="https://drive.google.com/file/d/1Gevpy9_YIumFMRDDkJTyQJjIT7V_o8iK/view?usp=sharing">Google Drive</a></td>
  </tr>
</table>


## Train and Evaluate Our Method
To train and evaluate our method on different datasets under desired granularity level, please follow the instructions [**here**](Command.md).

## Citation
If you find our work useful in your research, please consider citing:

    @article{yin2021transfgu,
      title={TransFGU: A Top-down Approach to Fine-Grained Unsupervised Semantic Segmentation},
      author={Zhaoyun, Yin and Pichao, Wang and Fan, Wang and Xianzhe, Xu and Hanling, Zhang and Hao, Li and Rong, Jin},
      journal={arXiv preprint arXiv:2112.01515},
      year={2021}
    }

## LICENSE
The code is released under the [MIT license](LICENSE).

## Copyright
Copyright (C) 2010-2021 Alibaba Group Holding Limited.