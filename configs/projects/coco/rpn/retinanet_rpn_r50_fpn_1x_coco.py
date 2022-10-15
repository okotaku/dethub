_base_ = [
    '../../../_base_/models/rpn/retinanet_rpn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=16)

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=True, base_batch_size=16)
