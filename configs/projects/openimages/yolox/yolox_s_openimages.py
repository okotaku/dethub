_base_ = [
    'mmdet::_base_/default_runtime.py', '../../../_base_/models/yolox_s.py',
    '../../../_base_/datasets/openimages/openimages_detection_yolox_640.py',
    '../../../_base_/schedules/yolox/yolox_100e.py'
]
custom_imports = dict(
    imports=['dethub', 'mmcls.models'], allow_failed_imports=False)
fp16 = dict(loss_scale='dynamic')

model = dict(bbox_head=dict(num_classes=601))

train_dataloader = dict(batch_size=32, num_workers=8)
val_dataloader = dict(batch_size=32, num_workers=8)
test_dataloader = val_dataloader

# runtime settings
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs={{_base_.num_last_epochs}},
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
