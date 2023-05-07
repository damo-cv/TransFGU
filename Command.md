## Evaluation and Visualization
#### To evaluate our method on COCO-Stuff-27:
```bash
CUDA_VISIBLE_DEVICES=0 python cocostuff_trainval.py -F eval/cocostuff_27 with eval_only=1 model.decoder.n_things=12 model.decoder.n_stuff=15 model.decoder.pretrained_weight=weight/trained/cocostuff_27_decoder.pt
```
#### To evaluate our method on COCO-Stuff-171:
```bash
CUDA_VISIBLE_DEVICES=0 python cocostuff_trainval.py -F eval/cocostuff_171 with eval_only=1 model.decoder.n_things=80 model.decoder.n_stuff=91 model.decoder.pretrained_weight=weight/trained/cocostuff_171_decoder.pt
```
#### To evaluate our method on COCO-80:
```bash
CUDA_VISIBLE_DEVICES=0 python coco_trainval.py -F eval/coco_80 with eval_only=1 model.decoder.n_things=80 model.decoder.pretrained_weight=weight/trained/coco_80_decoder.pt
```
#### To evaluate our method on Cityscapes:
```bash
CUDA_VISIBLE_DEVICES=0 python cityscapes_trainval.py -F eval/cityscapes with eval_only=1 model.decoder.n_things=27 model.decoder.pretrained_weight=weight/trained/cityscapes_decoder.pt
```
#### To evaluate our method on Pascal-VOC:
```bash
CUDA_VISIBLE_DEVICES=0 python pascalvoc_trainval.py -F eval/pascalvoc with eval_only=1 model.decoder.n_things=20 model.decoder.pretrained_weight=weight/trained/pascalvoc_decoder.pt
```
#### To evaluate our method on LIP-5:
```bash
CUDA_VISIBLE_DEVICES=0 python lip_trainval.py -F eval/lip_5 with eval_only=1 dataset.val_batch_size=16 model.decoder.n_things=5 model.decoder.pretrained_weight=weight/trained/lip_5_decoder.pt
```
#### To evaluate our method on LIP-16:
```bash
CUDA_VISIBLE_DEVICES=0 python lip_trainval.py -F eval/lip_16 with eval_only=1 dataset.val_batch_size=16 model.decoder.n_things=16 model.decoder.pretrained_weight=weight/trained/lip_16_decoder.pt
```
#### To evaluate our method on LIP-19:
```bash
CUDA_VISIBLE_DEVICES=0 python lip_trainval.py -F eval/lip_19 with eval_only=1 dataset.val_batch_size=16 model.decoder.n_things=19 model.decoder.pretrained_weight=weight/trained/lip_19_decoder.pt
```

## Training
#### For COCO-Stuff-27:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python cocostuff_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python cocostuff_generate_pseudo_label.py with dataset.train_batch_size=8 model.decoder.n_things=12 model.decoder.n_stuff=15
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python cocostuff_trainval.py -F train/cocostuff with dataset.num_workers=32 model.teacher_update_interval=2 model.bootstrapping_start_epoch=2 model.decoder.n_things=12 model.decoder.n_stuff=15
# To train the model (on the curated data)
CUDA_VISIBLE_DEVICES=0,1,2,3 python cocostuff_trainval.py -F train/cocostuff with dataset.num_workers=32 model.teacher_update_interval=1 model.bootstrapping_start_epoch=1 model.decoder.n_things=12 model.decoder.n_stuff=15 dataset.is_curated=1 lr=2.2e-4
```


#### For COCO-Stuff-171:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python cocostuff_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python cocostuff_generate_pseudo_label.py with dataset.train_batch_size=8 model.decoder.n_things=80 model.decoder.n_stuff=91
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python cocostuff_trainval.py -F train/cocostuff with dataset.num_workers=32 model.decoder.n_things=80 model.decoder.n_stuff=91
```

#### For COCO-80:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python cocostuff_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python cocostuff_generate_pseudo_label.py with dataset.train_batch_size=8 model.decoder.n_things=80 model.decoder.n_stuff=91
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python coco_trainval.py -F train/coco with dataset.num_workers=32 model.decoder.n_things=80
```

#### For Cityscapes:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python cityscapes_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python cityscapes_generate_pseudo_label.py with dataset.train_batch_size=2
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python cityscapes_trainval.py -F train/cityscapes with dataset.num_workers=32
```

#### For Pascal-VOC:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python pascalvoc_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python pascalvoc_generate_pseudo_label.py with dataset.train_batch_size=8
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python pascalvoc_trainval.py -F train/pascalvoc with dataset.num_workers=32
```

#### For LIP-5:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python lip_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python lip_generate_pseudo_label.py with dataset.train_batch_size=2 model.decoder.n_things=5
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python lip_trainval.py -F train/lip_5 with lr=1e-4 dataset.num_workers=32 dataset.resize=240 dataset.train_batch_size=512 model.teacher_update_interval=2 model.bootstrapping_start_epoch=2 model.decoder.n_things=5
```

#### For LIP-16:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python lip_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python lip_generate_pseudo_label.py with dataset.train_batch_size=2 model.decoder.n_things=16
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python lip_trainval.py -F train/lip_16 with dataset.num_workers=32 dataset.resize=240 dataset.train_batch_size=512 model.teacher_update_interval=2 model.bootstrapping_start_epoch=2 model.decoder.n_things=16
```

#### For LIP-19:
```bash
# Prepare class token feature bank
CUDA_VISIBLE_DEVICES=0 python lip_crop.py
# To generate pseudo mask labels
CUDA_VISIBLE_DEVICES=0 python lip_generate_pseudo_label.py with dataset.train_batch_size=2 model.decoder.n_things=19
# To train the model
CUDA_VISIBLE_DEVICES=0,1,2,3 python lip_trainval.py -F train/lip_19 with dataset.num_workers=32 model.decoder.n_things=19
```
