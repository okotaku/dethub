import torch
from mmcv.ops import bbox_overlaps as cudaext_bbox_overlaps

from mmdet.models.task_modules.assigners import BboxOverlaps2D as Base
from mmdet.models.task_modules.assigners.iou2d_calculator import \
    cast_tensor_type
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps as torch_bbox_overlaps
from mmdet.structures.bbox import get_box_tensor


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  is_aligned=False,
                  force_torch=False):
    if not force_torch and bboxes1.is_cuda and mode in ('iou', 'iof', 'giou'):
        return cudaext_bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
    else:
        return torch_bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


@TASK_UTILS.register_module(force=True)
class BboxOverlaps2D(Base):

    def __init__(self, scale=1., dtype=None, force_torch=False):
        self.scale = scale
        self.dtype = dtype
        self.force_torch = force_torch

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(
                bboxes1,
                bboxes2,
                mode,
                is_aligned,
                force_torch=self.force_torch)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(
            bboxes1, bboxes2, mode, is_aligned, force_torch=self.force_torch)
