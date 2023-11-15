_base_ = ['runtime_cfg.py']
# defaults to use registries in mmyolo
default_scope = 'mmyolo'

# configure default hooks
default_hooks = dict(
    param_scheduler=dict(type='YOLOv5ParamSchedulerHook',
                         scheduler_type='linear',
                         lr_factor=0.01,
                         max_epochs=100,
                         warmup_epochs=5,
                         warmup_bias_lr=0.1),
    # save checkpoint per epoch.
    checkpoint=dict(save_param_scheduler=False,
                    interval=1,
                    save_best='coco/bbox_mAP',
                    max_keep_ckpts=5),
    # validation results visualization, set True to enable it.
    visualization=dict(type='mmdet.DetVisualizationHook'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49)
]
visualizer = dict(type='mmdet.DetLocalVisualizer')
