# Prepare configs

For basic usage of configs, see [MMDetection Tutorial 1: Learn about Configs](https://mmdetection.readthedocs.io/en/stable/tutorials/config.html)

# Train a model

```
# single-gpu
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/train.py ${CONFIG_FILE}
# Example
$ docker compose exec dethub python /opt/site-packages/mmdet/.mim/tools/train.py configs/yolox_s.py

# multiple-gpu
$ docker compose exec dethub python -m torch.distributed.launch --nproc_per_node=${GPUS} /opt/site-packages/mmdet/.mim/tools/train.py ${CONFIG_FILE} --launcher pytorch
```

# Test a dataset

```
# single-gpu
$ mim test mmdet ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS}
# Example
$ mim test mmdet configs/yolox_s.py --checkpoint work_dirs/yolox_s/epoch_100.pth --out test.pkl --eval bbox
```

# More details

See [MMDetection Docs](https://mmdetection.readthedocs.io/en/stable/)
