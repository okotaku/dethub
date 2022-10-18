_base_ = [
    '../../../_base_/models/rpn/retinanet_rpn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=16)

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

auto_scale_lr = dict(enable=True, base_batch_size=16)
