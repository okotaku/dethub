_base_ = [
    'mmdet::_base_/default_runtime.py', '../../../_base_/models/yolox_x.py',
    '../../../_base_/datasets/coco_detection_yolox_ft_1536.py',
    '../../../_base_/schedules/yolox_50e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)

# model settings
num_classes = 8
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
data_root = 'data/livecell/'
metainfo = dict(
    CLASSES=[
        'shsy5y', 'a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'skov3', 'skbr3'
    ],
    PALETTE=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)])

train_dataset = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='livecell_coco_train_8class.json',
        data_prefix=dict(img='images/livecell_train_val_images/')))

train_dataloader = dict(batch_size=2, dataset=train_dataset)
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='livecell_coco_val_8class.json',
        data_prefix=dict(img='images/livecell_train_val_images/')))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoFastMetric',
    ann_file=data_root + 'livecell_coco_val_8class.json',
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
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(project='livecell', name='yolox_x_livecell'),
        define_metric_cfg={'coco/bbox_mAP': 'max'})
]
visualizer = dict(vis_backends=vis_backends)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
