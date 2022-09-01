from mmengine.config import Config

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX as Base


@DETECTORS.register_module(force=True)
class YOLOX(Base):
    arch_zoo = {
        **dict.fromkeys(['s'],
        dict(  # noqa
            backbone=dict(
                type='CSPDarknet',
                deepen_factor=0.33,
                widen_factor=0.5,
                out_indices=(2, 3, 4),
                use_depthwise=False,
                spp_kernal_sizes=(5, 9, 13),
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Swish'),
            ),
            neck=dict(
                type='YOLOXPAFPN',
                in_channels=[128, 256, 512],
                out_channels=128,
                num_csp_blocks=1,
                use_depthwise=False,
                upsample_cfg=dict(scale_factor=2, mode='nearest'),
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Swish')),
            bbox_head=dict(
                type='YOLOXHead',
                num_classes=80,
                in_channels=128,
                feat_channels=128,
                stacked_convs=2,
                strides=(8, 16, 32),
                use_depthwise=False,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Swish'),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='sum',
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='IoULoss',
                    mode='square',
                    eps=1e-16,
                    reduction='sum',
                    loss_weight=5.0),
                loss_obj=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='sum',
                    loss_weight=1.0),
                loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
            train_cfg=dict(
                assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
            test_cfg=dict(score_thr=0.01,
                          nms=dict(type='nms', iou_threshold=0.65))))
    }  # yapf: disable

    def __init__(self,
                 *args,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 arch=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            arch_settings = self.arch_zoo[arch]
            if backbone is not None:
                arch_settings['backbone'].update(backbone)
            if neck is not None:
                arch_settings['neck'].update(neck)
            if bbox_head is not None:
                arch_settings['bbox_head'].update(bbox_head)
            if train_cfg is not None:
                arch_settings['train_cfg'].update(train_cfg)
            if test_cfg is not None:
                arch_settings['train_cfg'].update(test_cfg)
            arch_settings = Config(arch_settings)
            backbone = arch_settings['backbone']
            neck = arch_settings['neck']
            bbox_head = arch_settings['bbox_head']
            train_cfg = arch_settings['train_cfg']
            test_cfg = arch_settings['test_cfg']
        else:
            assert backbone is not None, \
                'You should be set backbone if arch is None.'
            assert neck is not None, \
                'You should be set neck if arch is None.'
            assert bbox_head is not None, \
                'You should be set bbox_head if arch is None.'
        super(YOLOX, self).__init__(*args, backbone, neck, bbox_head,
                                    train_cfg, test_cfg, **kwargs)
