# PACO

> [PACO: Parts and Attributes of Common Objects](https://arxiv.org/abs/2301.01795)

<!-- [DATASET] -->

<div align=center>
<img src="https://github.com/facebookresearch/paco/blob/main/docs/teaser.png" height="300"/>
</div>

## Run demo

```
$ docker compose exec dethub python tools/image_demo.py configs/projects/paco/demo/000000000109.jpg configs/projects/paco/dino/dino-4scale_r50_paco_lvis.py https://github.com/okotaku/dethub-weights/releases/download/v0.1.1dino/dino-4scale_r50_paco_lvis-e460dbff.pth --out-file configs/projects/paco/demo/000000000109_demo.jpg --palette random
```

![plot](demo/000000000109_demo.jpg)

## Prepare datasets

1. Prepare LVIS dataset based on [README](../lvis/README.md)

2. Download data from [official repo](https://github.com/facebookresearch/paco).

```
wget https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
```

2. Unzip the files as follows

```
data/lvis_v1
├── annotations
|    ├── paco_lvis_v1_train.json
|    └── paco_lvis_v1_val.json
├── train2017
└── val2017
```

## Run train

Set env variables

```
$ export DATA_DIR=/path/to/data
```

Start a docker container

```
$ docker compose up -d dethub
```

Run train

```
# single gpu
$ docker compose exec dethub mim train mmdet configs/projects/paco/yolox/yolox_s_paco_lvis.py
# multi gpus
$ docker compose exec dethub mim train mmdet configs/projects/oxpacoford_pets/yolox/yolox_s_paco_lvis.py --gpus 2 --launcher pytorch
```
