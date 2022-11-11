# DetHub

[![build](https://github.com/okotaku/dethub/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/dethub/actions/workflows/build.yml)
[![license](https://img.shields.io/github/license/okotaku/dethub.svg)](https://github.com/okotaku/dethub/blob/main/LICENSE)

## Introduction

DetHub is an open source object detection / instance segmentation experiments hub. Our main contribution is supporting detection datasets and share baselines.

- Support more and more datasets
- Provide reproducible baseline configs for these datasets
- Provide pretrained models, results and inference codes for these datasets

Documentation: [docs](docs)

## Supported Datasets

- [x] [COCO](configs/projects/coco/)
- [x] [TensorFlow - Help Protect the Great Barrier Reef (Kaggle)](configs/projects/gbr_cots/)
- [x] [LIVECell dataset](configs/projects/livecell/)
- [x] [Sartorius - Cell Instance Segmentation (Kaggle)](configs/projects/sartorius_cellseg/)
- [x] [Vehicle Detection in Multi-Resolution Images (Solafune)](configs/projects/solafune_cardet/)
- [x] [LVIS](configs/projects/lvis/)
- [x] [CrowdHuman](configs/projects/crowdhuman/)

## Get Started

Please refer to [get_started.md](docs/source/get_started.md) for get started.
Other tutorials for:

- [run](docs/source/run.md)

## Contributing

### CONTRIBUTING

We appreciate all contributions to improve dethub. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This repo borrows the architecture design and part of the code from [mmdetection](https://github.com/open-mmlab/mmdetection).

Also, please check the following openmmlab projects and the corresponding Documentation.

- [OpenMMLab](https://openmmlab.com/)
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.

#### Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
