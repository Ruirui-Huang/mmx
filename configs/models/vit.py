# VITSEG Network
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
align_corners = False
model = dict(
    type='mmx.SegEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmx.VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        window_size=14,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        use_rel_pos=True,
        pretrain_use_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    neck=dict(
        type='mmx.SimpleFeaturePyramid',
        embed_dim=768,
        out_channels=256,
        in_index=3,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='mmx.USemHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        scale_factor=4,
        head_type="S",
        multicat=False,
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=0.4),
            dict(
                type='LovaszLoss',
                loss_name='loss_lovasz',
                reduction='none',
                loss_type='multi_class',
                loss_weight=0.6)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
                 avg_non_ignore=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))