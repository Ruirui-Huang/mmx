_base_ = ['runtime_cfg.py']
# defaults to use registries in mmseg
default_scope = 'mmseg'

# configure default hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(interval=1, save_best='mIoU', max_keep_ckpts=5),
    # validation results visualization, set True to enable it.
    visualization=dict(type='SegVisualizationHook'))

visualizer = dict(type='SegLocalVisualizer')