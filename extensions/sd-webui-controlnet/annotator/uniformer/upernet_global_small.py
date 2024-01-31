_base_ = [
    'configs/_base_/models/upernet_uniformer.py', 
    'configs/_base_/datasets/ade20k.py',
    'configs/_base_/default_runtime.py', 
    'configs/_base_/schedules/schedule_160k.py'
]

custom_imports = dict(
    imports=['annotator.uniformer.uniformer'],
    allow_failed_imports=False
)

model = dict(
    backbone=dict(
        type='UniFormer',
        embed_dim=[64, 128, 320, 512],
        layers=[3, 4, 8, 3],
        head_dim=64,
        drop_path_rate=0.25,
        windows=False,
        hybrid=False
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=150
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)