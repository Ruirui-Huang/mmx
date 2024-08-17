METAINFO = dict(
        classes=("background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant","stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"),
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],[0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192], [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],[0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192], [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

_base_ = [
    '../_base_/runtime_mmseg.py'
]

input_size = (512, 512)
base_lr = 0.01
val_interval = 2
max_epochs = 20
train_batch_size = 2
train_num_workers = 2
val_batch_size = 1
val_num_workers = 2
deepen_factor = 1.0
widen_factor = 1.0
arch = "P4S"
num_classes = len(METAINFO['classes'])

# dataset settings
dataset_type = 'BaseSegDataset'
data_root = 'D:/Data_pub/00_coco2017/'
run_name = 'v9_psp'
work_dir = 'D:/models/03_OS/Coco/' + run_name
pretrained = None

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    size=input_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomFlip', 
        prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Resize', 
        scale=input_size, 
        keep_ratio=False),
    dict(
        type='mmx.MaskImgPlot', img_save_path=work_dir + "/pipeline",
        save_img_num=10, 
        palette=METAINFO['palette']),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize', 
        scale=input_size, 
        keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/random_sampled_train2017', 
            seg_map_path='labels/random_sampled_train2017'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=val_num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/random_sampled_train2017', 
            seg_map_path='labels/random_sampled_train2017'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric', 
    iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Free-Efficient Network
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
act_cfg = dict(type='ReLU', inplace=True)

model = dict(
    type='mmx.EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    backbone=dict(
        type='mmyolo.YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels = 1024,
        deepen_factor=0.33,
        widen_factor=0.25,
        out_indices=(1, 2, 3, 4),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    decode_head=dict(
        type='mmx.DetPSPHead',
        in_channels=256,
        in_index=-1,
        channels=128,
        pool_scales=(1, 2, 4, 8),
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
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
        in_channels=128,
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

# optimizer
optimizer = dict(
    type='SGD', 
    lr=base_lr, 
    momentum=0.9, 
    weight_decay=0.0005)

optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=optimizer, 
    clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=2,
        by_epoch=False)
]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=2, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))