import numpy as np

from mmdet.datasets.transforms.transforms import MixUp as BaseMixUp
from mmdet.datasets.transforms.transforms import Mosaic as BaseMosaic
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class Mosaic(BaseMosaic):

    def __init__(self, *args, pad_val=114, **kwargs):
        if type(pad_val) == list or tuple:
            pad_val = np.array(pad_val)
        super().__init__(*args, pad_val=pad_val, **kwargs)


@TRANSFORMS.register_module(force=True)
class MixUp(BaseMixUp):

    def __init__(self, *args, pad_val=114, **kwargs):
        if type(pad_val) == list or tuple:
            pad_val = np.array(pad_val)
        super().__init__(*args, pad_val=pad_val, **kwargs)
