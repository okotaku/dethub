_base_ = [
    'mmdet::_base_/default_runtime.py',
    '../../../_base_/models/dino/dino-4scale_r50.py',
    '../../../_base_/datasets/objects365/objects365v2_detection_dino.py',
    '../../../_base_/schedules/dino/dino_12e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)
fp16 = dict(loss_scale=512.)

# model settings
num_classes = 365
model = dict(bbox_head=dict(num_classes=num_classes))
