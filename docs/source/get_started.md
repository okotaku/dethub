# Installation

Below are quick steps for installation:

```
pip install openmim
mim install mmcv==2.0.0rc4
pip install git+https://github.com/okotaku/dethub.git
```

# Inference

```
from mmengine.logging import print_log
from dethub.apis import DetInferencer

inferencer = DetInferencer('yolox_s_swin_s_coco', device='cuda:0')
inferencer(inputs='000000000025.jpg', print_result=True)

> {'predictions': [{'bboxes': [[381.07073974609375, 60.5645751953125, 604.7224731445312, 355.8541564941406],
[49.724632263183594, 353.8826904296875, 182.50323486328125, 415.73333740234375]], 'labels': [23, 23], 'scores':
[0.9441210627555847, 0.9047386646270752]}]}
{'predictions': [{'bboxes': [[381.07073974609375,
     60.5645751953125,
     604.7224731445312,
     355.8541564941406],
    [49.724632263183594,
     353.8826904296875,
     182.50323486328125,
     415.73333740234375]],
   'labels': [23, 23],
   'scores': [0.9441210627555847, 0.9047386646270752]}],
 'visualization': []}
```
