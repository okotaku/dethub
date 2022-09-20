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

Run demo

```
$ docker compose exec dethub python tools/image_demo.py ${IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUT_FILE}
# Example
$ docker compose exec dethub python tools/image_demo.py configs/projects/livecell/demo/A172_Phase_A7_1_00d00h00m_1.tif configs/projects/livecell/yolox/yolox_s_livecell.py https://github.com/okotaku/dethub-weights/releases/download/v0.0.1/yolox_s_livecell-b3f4347f.pth --out-file configs/projects/livecell/demo/A172_Phase_A7_1_00d00h00m_1_demo.jpg
```
