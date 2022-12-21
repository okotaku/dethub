from typing import Dict, List, Sequence

import numpy as np
from mmengine.logging import MMLogger

from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class CocoCountBboxL1Metric(CocoMetric):
    def __init__(self, *args,
                 count_thresholds=np.linspace(0.01, 0.99, num=99),
                 skip_map=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.count_thresholds = count_thresholds
        self.skip_map = skip_map

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            if 'bboxes' in pred:
                result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy())
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def eval_l1(self,
                         results: List[dict]) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.
        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_counts = [[] for _ in self.cat_ids]
        pred_counts = {thr:[[] for _ in self.cat_ids] for thr in self.count_thresholds}
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            ann_category_id = np.array([ann['category_id'] for ann in ann_info])
            pred_labels = np.array(results[i]['labels'])
            pred_scores = np.array(results[i]['scores'])
            for c in self.cat_ids:
                gt_counts[c].append(np.sum(ann_category_id == c))
                for thr in self.count_thresholds:
                    pred_labels_ = pred_labels[pred_scores > thr]
                    pred_counts[thr][c].append(np.sum(pred_labels_ == c))

        gt_counts = np.array(gt_counts)
        l1s = []
        for thr in self.count_thresholds:
            pred_counts_thr = np.array(pred_counts[thr])
            l1s.append(np.mean(np.abs(gt_counts - pred_counts_thr)))
        return l1s

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        if self.skip_map:
            eval_results = dict()
            # handle lazy init
            if self.cat_ids is None:
                self.cat_ids = self._coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['CLASSES'])
            if self.img_ids is None:
                self.img_ids = self._coco_api.get_img_ids()
        else:
            eval_results = super().compute_metrics(results)
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        _, preds = zip(*results)

        l1s = self.eval_l1(preds)
        log_msg = []
        for l1, thr in zip(l1s, self.count_thresholds):
            eval_results[f'l1_{thr}'] = l1
        best_l1 = np.min(l1s)
        eval_results['l1'] = best_l1
        eval_results['best_thr'] = self.count_thresholds[np.argmin(l1s)]
        log_msg.append(f'\nl1\t{best_l1:.4f}')
        log_msg.append(f'\nbest_thr\t{self.count_thresholds[np.argmin(l1s)]}')
        log_msg = ''.join(log_msg)
        logger.info(log_msg)

        return eval_results
