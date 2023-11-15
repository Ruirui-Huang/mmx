_base_ = ['runtime_cfg.py']

# defaults to use registries in mmcls
default_scope = 'mmcls'

# configure default hooks
# validation results visualization, set True to enable it.
default_hooks = dict(checkpoint=dict(type='CheckpointHook',
                                     interval=1,
                                     save_best='accuracy/top1',
                                     max_keep_ckpts=5),
                     visualization=dict(type='VisualizationHook', enable=False))

visualizer = dict(type='ClsVisualizer')