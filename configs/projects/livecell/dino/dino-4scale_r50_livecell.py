_base_ = [
    'mmdet::_base_/default_runtime.py',
    '../../../_base_/models/dino/dino-4scale_r50.py',
    '../../../_base_/datasets/livecell/livecell_detection_dino.py',
    '../../../_base_/schedules/dino/dino_36e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)

# model settings
num_classes = 8
model = dict(
    num_queries=3000,
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(max_per_img=3000))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'  # noqa
