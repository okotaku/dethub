# dataset settings
img_scale = (640, 640)  # height, width

dataset_type = 'OpenImagesDataset'
data_root = 'data/OpenImages/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/oidv6-train-annotations-bbox.csv',
        data_prefix=dict(img='OpenImages/train/'),
        label_file='annotations/class-descriptions-boxable.csv',
        hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
        meta_file='annotations/train-image-metas.pkl',
        pipeline=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    pipeline=train_pipeline)

val_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances', 'image_level_labels'))
]

demo_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/validation-annotations-bbox.csv',
        data_prefix=dict(img='OpenImages/validation/'),
        label_file='annotations/class-descriptions-boxable.csv',
        hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
        meta_file='annotations/validation-image-metas.pkl',
        image_level_ann_file='annotations/validation-'
        'annotations-human-imagelabels-boxable.csv',
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/validation-annotations-bbox.csv',
        data_prefix=dict(img='OpenImages/validation/'),
        label_file='annotations/class-descriptions-boxable.csv',
        hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
        meta_file='annotations/validation-image-metas.pkl',
        image_level_ann_file='annotations/validation-'
        'annotations-human-imagelabels-boxable.csv',
        # pipeline=val_pipeline
        pipeline=demo_pipeline))

val_evaluator = dict(
    type='OpenImagesMetric',
    iou_thrs=0.5,
    ioa_thrs=0.5,
    use_group_of=True,
    get_supercategory=True)
test_evaluator = val_evaluator
