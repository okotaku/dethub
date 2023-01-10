_base_ = [
    'mmdet::_base_/default_runtime.py', '../../../_base_/models/yolox_s.py',
    '../../../_base_/datasets/lvis/lvis_detection_yolox_640.py',
    '../../../_base_/schedules/yolox/yolox_300e.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)

# model settings
num_classes = 1203
model = dict(
    bbox_head=dict(num_classes=num_classes), test_cfg=dict(score_thr=0.001))
