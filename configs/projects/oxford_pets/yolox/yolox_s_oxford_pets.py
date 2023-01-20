_base_ = [
    'mmdet::_base_/default_runtime.py', '../../../_base_/models/yolox_s.py',
    '../../../_base_/datasets/coco/coco_detection_yolox_ft_640.py',
    '../../../_base_/schedules/yolox/yolox_50e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)

# model settings
num_classes = 37
model = dict(bbox_head=dict(num_classes=num_classes))

# dataset settings
data_root = 'data/oxford-pets/'
metainfo = dict(
    classes=[
        'cat-Abyssinian', 'cat-Bengal', 'cat-Birman', 'cat-Bombay',
        'cat-British_Shorthair', 'cat-Egyptian_Mau', 'cat-Maine_Coon',
        'cat-Persian', 'cat-Ragdoll', 'cat-Russian_Blue', 'cat-Siamese',
        'cat-Sphynx', 'dog-american_bulldog', 'dog-american_pit_bull_terrier',
        'dog-basset_hound', 'dog-beagle', 'dog-boxer', 'dog-chihuahua',
        'dog-english_cocker_spaniel', 'dog-english_setter',
        'dog-german_shorthaired', 'dog-great_pyrenees', 'dog-havanese',
        'dog-japanese_chin', 'dog-keeshond', 'dog-leonberger',
        'dog-miniature_pinscher', 'dog-newfoundland', 'dog-pomeranian',
        'dog-pug', 'dog-saint_bernard', 'dog-samoyed', 'dog-scottish_terrier',
        'dog-shiba_inu', 'dog-staffordshire_bull_terrier',
        'dog-wheaten_terrier', 'dog-yorkshire_terrier'
    ],
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
             (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
             (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
             (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
             (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
             (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
             (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
             (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
             (134, 134, 103)])

train_dataset = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))

train_dataloader = dict(batch_size=32, dataset=train_dataset)
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# training settings
base_lr = 0.05
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# learning rate
max_epochs = 50
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
        max_keep_ckpts=3,  # only keep latest 3 checkpoints
        rule='greater'),
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

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
