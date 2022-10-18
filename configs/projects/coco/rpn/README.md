# RPN

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143973617-387c7561-82f4-40b2-b78e-4776394b1b8b.png"/>
</div>

## Results and Models

|     Backbone     | AR@100 | AR@300 | AR@1000 |            Config             |                                                 Download                                                  |
| :--------------: | :----: | :---------------------------: | :-------------------------------------------------------------------------------------------------------: |
| mmdet rpn_r50_fpn_1x_coco |  45.50  |  52.89  |  58.19  | [config](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/rpn/rpn_r50_fpn_1x_coco.py) | - |
| mmdet ga_rpn_r50_fpn_1x_coco |  59.08  |  65.10  |  68.36  | [config](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/guided_anchoring/ga-rpn_r50-caffe_fpn_1x_coco.py) | - |
| atss_rpn_r50_fpn_1x_coco |  60.81  |  63.46  |  64.41  | [config](atss_rpn_r50_fpn_1x_coco.py) | [model](<>) |
| fcos_rpn_r50_fpn_1x_coco |  59.60  |  62.30  |  63.38  | [config](fcos_rpn_r50_fpn_1x_coco.py) | [model](<>) |
| retinanet_rpn_r50_fpn_1x_coco |  55.98  |  59.08  |  60.33  | [config](retinanet_rpn_r50_fpn_1x_coco.py) | [model](<>) |
| tood_rpn_r50_fpn_1x_coco |  62.94  |  65.70  |  66.43  | [config](tood_rpn_r50_fpn_1x_coco.py) | [model](<>) |

## Citation

```latex
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  year={2015}
}
```
