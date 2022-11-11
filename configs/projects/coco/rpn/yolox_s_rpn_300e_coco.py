_base_ = [
    '../../../_base_/models/rpn/yolox_s_rpn.py',
    '../../../_base_/datasets/coco/coco_detection_yolox_640.py',
    '../../../_base_/schedules/yolox/yolox_300e.py',
    'mmdet::_base_/default_runtime.py'
]
fp16 = dict(loss_scale=512.)

train_dataloader = dict(batch_size=32)

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

# runtime settings
default_hooks = dict(
    checkpoint=dict(
        save_best='auto',
        interval={{_base_.interval}},
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ),
    visualization=dict(draw=False, interval=5))
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs={{_base_.num_last_epochs}},
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

auto_scale_lr = dict(enable=True, base_batch_size=64)
