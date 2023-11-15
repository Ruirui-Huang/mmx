# Free-Efficient Network
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
act_cfg = dict(type='ReLU', inplace=True)
model = dict(
    type='mmx.SegEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmx.CSPFRegNet',
        arch="P4B",
        deepen_factor=1.0,
        widen_factor=1.0,
        group_width=8,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        norm_eval=False,
        stem_large_kernel_size=True),
    neck=dict(
        type='mmx.FEINetUpHead',
        arch="P4B",
        widen_factor=1.0,
        head_type="S",
        multicat=False,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    decode_head=dict(
        type='mmx.SegFCNHead',
        scale_factor=4,
        in_channels=160,
        in_index=3,
        channels=64,
        kernel_size=3,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.2,
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.4,
                avg_non_ignore=True),
            dict(
                type='LovaszLoss',
                loss_name='loss_lovasz',
                reduction='none',
                loss_type='multi_class',
                loss_weight=0.6)
        ]),
    auxiliary_head=dict(
        type='STDCHead',
        in_channels=32,
        in_index=2,
        channels=64,
        kernel_size=3,
        num_convs=1,
        concat_input=False,
        boundary_threshold=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=True,
                loss_weight=0.4,
                avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.1)
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))