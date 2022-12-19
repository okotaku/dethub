_base_ = [
    'mmdet::_base_/default_runtime.py',
    '../../../_base_/models/rtmdet/rtmdet_s_swin_s.py',
    '../../../_base_/datasets/coco/coco_detection_rtmdet_640_imagenet.py',
    '../../../_base_/schedules/rtmdet/rtmdet_100e_adamw.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale='dynamic')

train_dataloader = dict(batch_size=32, num_workers=8)
val_dataloader = dict(batch_size=32, num_workers=8)
test_dataloader = val_dataloader

# training settings
max_epochs = 100
stage2_num_epochs = 10
interval = 10

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}})
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
