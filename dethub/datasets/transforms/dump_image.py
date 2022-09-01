from os import path as osp

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData
from torch.multiprocessing import Value

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer


@TRANSFORMS.register_module()
class DumpImage:
    """Dump the image processed by the pipeline.

    Args:
        max_imgs (int): Maximum value of output.
        dump_dir (str): Dump output directory.
    """

    def __init__(self, max_imgs, dump_dir):
        self.max_imgs = max_imgs
        self.dump_dir = dump_dir
        mmengine.mkdir_or_exist(self.dump_dir)
        self.num_dumped_imgs = Value('i', 0)
        self.det_local_visualizer = DetLocalVisualizer()

    def __call__(self, results):
        """Dump the input image to the specified directory.

        No changes will be
        made.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            results (dict): Result dict from loading pipeline. (same as input)
        """

        enable_dump = False
        with self.num_dumped_imgs.get_lock():
            if self.num_dumped_imgs.value < self.max_imgs:
                self.num_dumped_imgs.value += 1
                enable_dump = True
                dump_id = self.num_dumped_imgs.value

        if enable_dump:
            img = results['img']
            out_file = osp.join(self.dump_dir, f'dump_{dump_id}.png')
            mmcv.imwrite(img.astype(np.uint8), out_file)

            gt_instances = InstanceData()
            gt_instances.bboxes = torch.Tensor(results['gt_bboxes'].tensor)
            gt_instances.labels = torch.Tensor(
                results['gt_bboxes_labels']).long()
            if 'gt_masks' in results:
                gt_instances.masks = results['gt_masks'].masks

            gt_det_data_sample = DetDataSample()
            gt_det_data_sample.gt_instances = gt_instances
            out_file = osp.join(self.dump_dir, f'dump_{dump_id}_withgt.png')
            self.det_local_visualizer.add_datasample(
                'image', img, gt_det_data_sample, out_file=out_file)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + (f'(max_imgs={self.max_imgs}, '
                                              f'dump_dir="{self.dump_dir}")')

        return repr_str
