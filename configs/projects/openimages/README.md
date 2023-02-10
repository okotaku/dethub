# Open Images Dataset

> [Open Images Dataset](https://arxiv.org/abs/1811.00982)

## Run demo

```
$ docker compose exec dethub python tools/image_demo.py configs/projects/openimages/demo/000002b66c9c498e.jpg configs/projects/openimages/yolox/yolox_s_openimages.py --weights https://github.com/okotaku/dethub-weights/releases/download/v0.1.1openimages_yolox/yolox_s_openimages-46accb21.pth --out-dir configs/projects/openimages/demo/result
```

![plot](demo/000000000025_demo.jpg)

## Prepare datasets

Prease refer to [mmdet page](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/openimages/README.md)

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
$ docker compose exec dethub mim train mmdet configs/projects/openimages/yolox/yolox_s_openimages.py
# multi gpus
$ docker compose exec dethub mim train mmdet configs/projects/openimages/yolox/yolox_s_openimages.py --gpus 2 --launcher pytorch
```
