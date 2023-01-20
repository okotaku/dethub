_base_ = [
    'mmdet::_base_/default_runtime.py', '../../../_base_/models/yolox_x.py',
    '../../../_base_/datasets/coco/coco_detection_yolox_ft_1536.py',
    '../../../_base_/schedules/yolox/yolox_70e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)

# model settings
num_classes = 3
model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(1280, 1792),
                size_divisor=32,
                interval=10)
        ]),
    bbox_head=dict(num_classes=num_classes))

# dataset settings
data_root = 'data/sartorius_cellseg/'
metainfo = dict(
    classes=['shsy5y', 'astro', 'cort'],
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142)])

train_dataset = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='dtrain.json',
        data_prefix=dict(img='')))

train_dataloader = dict(batch_size=2, dataset=train_dataset)
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='dval.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoFastMetric',
    ann_file=data_root + 'dval.json',
    metric='bbox',
    proposal_nums=(100, 300, 3000))
test_evaluator = val_evaluator

# optimizer
base_lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# learning rate
max_epochs = 70
num_last_epochs = 10
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to -num_last_epochs epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last -num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

# runtime settings
default_hooks = dict(
    checkpoint=dict(
        save_best='auto',
        interval={{_base_.interval}},
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ),
    visualization=dict(draw=False, interval=1))
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
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='sartorius_cellseg', name='yolox_x_sartorius_cellseg'),
        define_metric_cfg={'coco/bbox_mAP': 'max'})
]
visualizer = dict(vis_backends=vis_backends)

load_from = 'https://github.com/okotaku/dethub-weights/releases/download/v0.0.1/yolox_x_livecell-b1fb7170.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
