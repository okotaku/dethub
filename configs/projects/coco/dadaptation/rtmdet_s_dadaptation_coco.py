_base_ = [
    'mmdet::_base_/default_runtime.py',
    '../../../_base_/models/rtmdet/rtmdet_s.py',
    '../../../_base_/datasets/coco/coco_detection_rtmdet_s_640.py',
    '../../../_base_/schedules/dadaptation/rtmdet_300e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)

train_dataloader = dict(batch_size=128, num_workers=8)

# runtime settings
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline={{_base_.train_pipeline_stage2}})
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=256)
