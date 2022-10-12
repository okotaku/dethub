# training settings
max_epochs = 20
num_last_epochs = 10
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 3 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 3 to -num_last_epochs epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=3,
        T_max=max_epochs - 5,
        end=max_epochs - 5,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last -num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - 5,
        end=max_epochs,
    )
]
