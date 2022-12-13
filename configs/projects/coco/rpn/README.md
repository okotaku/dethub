# RPN

[model page](../../../rpn/README.md)

## Results and Models

|           Backbone            | AR@100 | AR@300 | AR@1000 |                                   Config                                    |                                   Download                                    |
| :---------------------------: | :----: | :----: | :-----: | :-------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
|   mmdet rpn_r50_fpn_1x_coco   | 45.50  | 52.89  |  58.19  | [config](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/rpn/rpn_r50_fpn_1x_coco.py) |                                       -                                       |
| mmdet ga_rpn_r50_fpn_1x_coco  | 59.08  | 65.10  |  68.36  | [config](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/guided_anchoring/ga-rpn_r50-caffe_fpn_1x_coco.py) |                                       -                                       |
|   atss_rpn_r50_fpn_1x_coco    | 60.81  | 63.46  |  64.41  |                    [config](atss_rpn_r50_fpn_1x_coco.py)                    | [model](https://github.com/okotaku/dethub-weights/releases/download/v0.1.1cocorpn/atss_rpn_r50_fpn_1x_coco-81af958b.pth) |
|   fcos_rpn_r50_fpn_1x_coco    | 59.60  | 62.30  |  63.38  |                    [config](fcos_rpn_r50_fpn_1x_coco.py)                    | [model](https://github.com/okotaku/dethub-weights/releases/download/v0.1.1cocorpn/fcos_rpn_r50_fpn_1x_coco-b44310f1.pth) |
| retinanet_rpn_r50_fpn_1x_coco | 55.98  | 59.08  |  60.33  |                 [config](retinanet_rpn_r50_fpn_1x_coco.py)                  | [model](https://github.com/okotaku/dethub-weights/releases/download/v0.1.1cocorpn/retinanet_rpn_r50_fpn_1x_coco-b459621a.pth) |
|   tood_rpn_r50_fpn_1x_coco    | 62.94  | 65.70  |  66.43  |                    [config](tood_rpn_r50_fpn_1x_coco.py)                    | [model](https://github.com/okotaku/dethub-weights/releases/download/v0.1.1cocorpn/tood_rpn_r50_fpn_1x_coco-69402644.pth) |
|     yolox_s_rpn_300e_coco     | 60.08  | 62.16  |  63.14  |                     [config](yolox_s_rpn_300e_coco.py)                      | [model](https://github.com/okotaku/dethub-weights/releases/download/v0.1.1cocorpn/yolox_s_rpn_300e_coco-e6d942f1.pth) |
