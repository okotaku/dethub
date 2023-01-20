_base_ = [
    'mmdet::_base_/default_runtime.py',
    '../../../_base_/models/rtmdet_x.py',
    '../../../_base_/datasets/solafune_cardet/solafune_cardet_rtmdet_1280.py',  # noqa
    '../../../_base_/schedules/rtmdet/rtmdet_100e.py',
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale='dynamic')

num_classes = 1
model = dict(bbox_head=dict(num_classes=num_classes))

# dataset settings
data_root = 'data/solafune-cardet/'
metainfo = dict(classes=['car'], palette=[(220, 20, 60)])

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='dtrain_fold0.json',
        data_prefix=dict(img='train_images/')))
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='dval_fold0.json',
        data_prefix=dict(img='train_images/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'dval_fold0.json')
test_evaluator = val_evaluator

# optimizer
max_epochs = 100
stage2_num_epochs = 10
base_lr = 0.006
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

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
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='solafune_cardet', name='rtmdet_x_solafune_cardet'),
        define_metric_cfg={'coco/bbox_mAP': 'max'})
]
visualizer = dict(vis_backends=vis_backends)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (2 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
