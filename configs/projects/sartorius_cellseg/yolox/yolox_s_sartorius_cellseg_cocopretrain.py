_base_ = [
    'mmdet::_base_/default_runtime.py', '../../../_base_/models/yolox_s.py',
    '../../../_base_/datasets/coco/coco_detection_yolox_ft_1536.py',
    '../../../_base_/schedules/yolox/yolox_100e.py'
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
                random_size_range=(1024, 2048),
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

train_dataloader = dict(batch_size=4, dataset=train_dataset)
val_dataloader = dict(
    batch_size=4,
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
            project='sartorius_cellseg',
            name='yolox_s_sartorius_cellseg_cocopretrain'),
        define_metric_cfg={'coco/bbox_mAP': 'max'})
]
visualizer = dict(vis_backends=vis_backends)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
