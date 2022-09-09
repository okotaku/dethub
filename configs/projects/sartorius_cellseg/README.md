# Sartorius - Cell Instance Segmentation (Kaggle)

Kaggle [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

## Prepare datasets

1. Download competition data from Kaggle

```
kaggle competitions download -c sartorius-cell-instance-segmentation
```

2. Download coco format json.

```
kaggle datasets download https://www.kaggle.com/datasets/takuok/sartorius-cocoformat
```

\*We prepared coco format files from [this script](../../../tools/dataset_converters/prepare_sartorius_cellseg.py).

3. Unzip the files as follows

```
├── train
├── train_semi_supervised
├── train.csv
├── dtrain.json
└── dval.json
```

## Run train

Set env variables

```
$ export DATA_DIR="/path/to/data"
```

Start a docker container

```
$ docker compose up -d dethub
```

Run train

```
# single gpu
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/train.py configs/projects/sartorius_cellseg/yolox/yolox_s_sartorius_cellseg.py
# multi gpus
$ docker compose exec dethub python -m torch.distributed.launch --nproc_per_node=2 /opt/site-packages/mmdet/.mim/tools/train.py configs/projects/sartorius_cellseg/yolox/yolox_s_sartorius_cellseg.py --launcher pytorch
```

## Acknowledgement

[Kaggle Sartorius - Cell Instance Segmentation 1st place solution](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution)
