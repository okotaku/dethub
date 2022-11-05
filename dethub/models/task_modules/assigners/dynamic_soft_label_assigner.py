from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import \
    DynamicSoftLabelAssigner as Base
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.registry import TASK_UTILS

try:
    import ops
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please compile ops.')

INF = 100000000
EPS = 1.0e-7


def get_in_gt_and_in_center_info(priors, gt_bboxes):
    """Use CUDA extension to avoid unnecessary memory allocation."""
    if priors.is_cuda:
        num_prior = priors.size(0)
        num_gt = gt_bboxes.size(0)
        is_in_gts = torch.empty(
            num_prior, num_gt, device=priors.device, dtype=torch.bool)
        ops.check_prior_in_gt_dsla(priors, gt_bboxes, is_in_gts)
    else:
        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
    valid_mask = is_in_gts.sum(dim=1) > 0
    return valid_mask, is_in_gts


@TASK_UTILS.register_module(force=True)
class DynamicSoftLabelAssigner(Base):

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)

        valid_mask, is_in_gts = get_in_gt_and_in_center_info(priors, gt_bboxes)

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = torch.empty(
            valid_pred_scores.size(0),
            valid_pred_scores.size(1),
            device=valid_pred_scores.device,
            dtype=valid_pred_scores.dtype)
        for i in range(valid_pred_scores.size(2)):
            soft_cls_cost = soft_cls_cost + F.binary_cross_entropy_with_logits(
                valid_pred_scores[..., i],
                soft_label[..., i],
                reduction='none') * scale_factor[..., i].abs().pow(2.0)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
