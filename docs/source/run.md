# Data Preparation

Prepare datasets in data directory. You can reference each datasets format on [each projects README](../../configs/projects).

```
/path/to/data
├── coco
├── gbr_cots
├── livecell
└── sartorius_cellseg
```

# Environment setup

Clone repo

```
$ git clone https://github.com/okotaku/dethub
```

Set env variables

```
$ export DATA_DIR=/path/to/data
```

Start a docker container

```
$ docker compose up -d dethub
# optional install
$ docker compose exec dethub pip install -r docker/dev.txt
```

# Prepare configs

For basic usage of configs, see [MMDetection Tutorial 1: Learn about Configs](https://mmdetection.readthedocs.io/en/stable/tutorials/config.html)

# Train a model

```
# single-gpu
$ docker compose exec dethub mim train mmdet ${CONFIG_FILE}
# Example
$ docker compose exec dethub mim train mmdet configs/projects/livecell/yolox/yolox_s_livecell.py

# multiple-gpu
$ docker compose exec dethub mim train mmdet ${CONFIG_FILE} --gpus ${GPUS} --launcher pytorch
```

# Test a dataset

```
# single-gpu
$ docker compose exec dethub mim test mmdet ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE}
# Example
$ docker compose exec dethub mim test mmdet configs/projects/livecell/yolox/yolox_s_livecell.py --checkpoint work_dirs/yolox_s_livecell/epoch_100.pth
```

# Run demo

```
$ docker compose exec dethub python tools/image_demo.py ${IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUT_FILE}
# Example
$ docker compose exec dethub python tools/image_demo.py configs/projects/livecell/demo/A172_Phase_A7_1_00d00h00m_1.tif configs/projects/livecell/yolox/yolox_s_livecell.py --weights work_dirs/yolox_s_livecell/epoch_100.pth --out-dir configs/projects/livecell/demo/result
```

# More details

See [MMDetection Docs](https://mmdetection.readthedocs.io/en/3.x/)
