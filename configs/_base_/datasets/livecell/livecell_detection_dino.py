# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/livecell/'
metainfo = dict(
    classes=[
        'shsy5y', 'a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'skov3', 'skbr3'
    ],
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)])
file_client_args = dict(backend='disk')

multi = 1536 / 1333
plus = 400

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (int((s[0] + plus) * multi), int(s[1] * multi))
                        for s in [(480, 1333), (512, 1333), (
                            544, 1333), (576, 1333), (608, 1333), (
                                640, 1333), (672, 1333), (
                                    704, 1333), (736, 1333), (768,
                                                              1333), (800,
                                                                      1333)]
                    ],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(int((s[0] + plus) * multi), int(s[1] * multi))
                            for s in [(400, 4200), (500, 4200), (600, 4200)]],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(int(
                        (384 + plus * 600 / 1333) * multi), int(600 * multi)),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (int((s[0] + plus) * multi), int(s[1] * multi))
                        for s in [(480, 1333), (512, 1333), (
                            544, 1333), (576, 1333), (608, 1333), (
                                640, 1333), (672, 1333), (
                                    704, 1333), (736, 1333), (768,
                                                              1333), (800,
                                                                      1333)]
                    ],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='Resize',
        scale=(int(1333 * multi), int((800 + plus) * multi)),
        keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='livecell_coco_train_8class.json',
        data_prefix=dict(img='images/livecell_train_val_images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='livecell_coco_val_8class.json',
        data_prefix=dict(img='images/livecell_train_val_images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoFastMetric',
    ann_file=data_root + 'livecell_coco_val_8class.json',
    metric='bbox',
    proposal_nums=(100, 300, 3000))
test_evaluator = val_evaluator
