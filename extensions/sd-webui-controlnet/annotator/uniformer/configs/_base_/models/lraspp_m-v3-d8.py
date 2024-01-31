# model settings
norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetV3',
        arch='large',
        out_indices=(1, 3, 16),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 24, 960),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
