# CrowdHuman

> [CrowdHuman: A Benchmark for Detecting Human in a Crowd](https://arxiv.org/abs/1805.00123)

<!-- [DATASET] -->

## Abstract

Human detection has witnessed impressive progress in recent years. However, the occlusion issue of detecting human in highly crowded environments is far from solved. To make matters worse, crowd scenarios are still under-represented in current human detection benchmarks. In this paper, we introduce a new dataset, called CrowdHuman, to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. There are a total of 470K human instances from the train and validation subsets, and  22.6 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. Baseline performance of state-of-the-art detection frameworks on CrowdHuman is presented. The cross-dataset generalization results of CrowdHuman dataset demonstrate state-of-the-art performance on previous dataset including Caltech-USA, CityPersons, and Brainwash without bells and whistles. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/201292160-6905c346-7882-4236-8ce1-ea7d50eea932.png" height="300"/>
</div>

## Run demo

```
$ docker compose exec dethub python tools/image_demo.py configs/projects/crowdhuman/demo/273271,106220007be3af91.jpg configs/projects/crowdhuman/yolox/yolox_s_crowdhuman.py https://github.com/okotaku/dethub-weights/releases/download/v0.1.1crowdhuman/yolox_s_crowdhuman-fd5a218a.pth --out-file configs/projects/crowdhuman/demo/273271,106220007be3af91_demo.jpg
```

![plot](demo/273271,106220007be3af91_demo.jpg)

## Prepare datasets

1. Download data from from [official page](https://www.crowdhuman.org/)

2. Unzip the files as follows

```
data/CrowdHuman
├── annotation_train.odgt
├── annotation_val.odgt
├── id_hw_train.json
├── id_hw_val.json
└── Images
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
$ docker compose exec dethub mim train mmdet configs/projects/crowdhuman/yolox/yolox_s_crowdhuman.py
# multi gpus
$ docker compose exec dethub mim train mmdet configs/projects/crowdhuman/yolox/yolox_s_crowdhuman.py --gpus 2 --launcher pytorch
```

## Citation

```latex
@article{shao2018crowdhuman,
  title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
  author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:1805.00123},
  year={2018}
}
```
