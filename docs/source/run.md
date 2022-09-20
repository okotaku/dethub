# Prepare configs

For basic usage of configs, see [MMDetection Tutorial 1: Learn about Configs](https://mmdetection.readthedocs.io/en/stable/tutorials/config.html)

# Train a model

```
# single-gpu
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/train.py ${CONFIG_FILE}
# Example
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/train.py configs/projects/livecell/yolox/yolox_s_livecell.py

# multiple-gpu
$ docker compose exec dethub python -m torch.distributed.launch --nproc_per_node=${GPUS} /opt/site-packages/mmdet/.mim/tools/train.py ${CONFIG_FILE} --launcher pytorch
```

# Test a dataset

```
# single-gpu
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
# Example
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/test.py configs/projects/livecell/yolox/yolox_s_livecell.py work_dirs/yolox_s_livecell/epoch_100.pth
```

# Run demo

```
$ docker compose exec dethub python tools/image_demo.py ${IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUT_FILE}
# Example
$ docker compose exec dethub python tools/image_demo.py configs/projects/livecell/demo/A172_Phase_A7_1_00d00h00m_1.tif configs/projects/livecell/yolox/yolox_s_livecell.py work_dirs/yolox_s_livecell/epoch_100.pth --out-file configs/projects/livecell/demo/A172_Phase_A7_1_00d00h00m_1_demo.jpg
```

# More details

See [MMDetection Docs](https://mmdetection.readthedocs.io/en/stable/)
