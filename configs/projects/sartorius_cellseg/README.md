# Sartorius - Cell Instance Segmentation

Kaggle [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

## Prepare datasets

1. Download competition data from Kaggle

```
kaggle competitions download -c sartorius-cell-instance-segmentation
```

2. Download coco format json.

We prepared coco format files from [this script](../../../tools/dataset_converters/prepare_sartorius_cellseg.py).

```
kaggle competitions download -c sartorius-cell-instance-segmentation
kaggle datasets download https://www.kaggle.com/datasets/takuok/sartorius-cocoformat
```

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
$ docker compose exec dethub mim train mmdet yolox_s_sartorius_cellseg.py --gpus 2 --launcher pytorch
```
