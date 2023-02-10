_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    '../../../_base_/schedules/dadaptation/dadaptation_1x.py',
    'mmdet::_base_/default_runtime.py'
]
custom_imports = dict(imports=['dethub'], allow_failed_imports=False)

train_dataloader = dict(batch_size=8)
