# Prepare configs

For basic usage of configs, see [MMDetection Tutorial 1: Learn about Configs](https://mmdetection.readthedocs.io/en/stable/tutorials/config.html)

# Train a model

```
# single-gpu
$ docker compose exec dethub mim train mmdet ${CONFIG_FILE} --gpus 1
# Example
$ docker compose exec dethub mim train mmdet configs/yolox_s.py --gpus 1

# multiple-gpu
$ docker compose exec dethub mim train mmdet ${CONFIG_FILE} --gpus ${GPUS} --launcher pytorch
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
