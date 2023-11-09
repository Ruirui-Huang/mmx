# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import os.path as osp
from importlib import import_module

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# from mmseg.utils import register_all_modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description='Visual task test (and eval) a model')
    # parser.add_argument('cvlib',
    #                     help='computer vision library',
    #                     choices=['mmcls', 'mmdet', 'mmseg', 'mmyolo'])
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir',
                        help=('if specified, the evaluation metric results will be dumped'
                              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--json-prefix',
        type=str,
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved. '
                        'If specified, it will be automatically saved '
                        'to the work_dir/timestamp/show_dir')
    parser.add_argument('--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visulizer = cfg.visualizer
            visulizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError('VisualizationHook must be included in default_hooks.'
                           'refer to usage '
                           '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

    
def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])


    cfg.load_from = args.checkpoint
    try:
        for custom_hook in cfg.custom_hooks:
            if custom_hook["type"] == "EMAHook":
                cfg.custom_hooks.remove(custom_hook)
    except: pass

    for vis_backend in cfg.visualizer["vis_backends"]:
        if "MLflowVisBackend" in vis_backend["type"]:
            cfg.visualizer["vis_backends"].remove(vis_backend)
        
    if cfg.default_scope == 'mmyolo':
        if args.show or args.show_dir:
            from mmdet.engine.hooks.utils import trigger_visualization_hook
            cfg = trigger_visualization_hook(cfg, args)

        # add `format_only` and `outfile_prefix` into cfg
        if args.json_prefix is not None:
            cfg_json = {
                'test_evaluator.format_only': True,
                'test_evaluator.outfile_prefix': args.json_prefix
            }
            cfg.merge_from_dict(cfg_json)
        # Determine whether the custom metainfo fields are all lowercase
        from mmyolo.utils import is_metainfo_lower
        is_metainfo_lower(cfg)
    
    # if args.tta:
    #     assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
    #                                " Can't use tta !"
    #     assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
    #                                   "in config. Can't use tta !"

    #     cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    #     cfg.tta_model.module = cfg.model
    #     cfg.model = cfg.tta_model
    #     if cfg.default_scope == 'mmyolo':
    #         test_data_cfg = cfg.test_dataloader.dataset
    #         while 'dataset' in test_data_cfg:
    #             test_data_cfg = test_data_cfg['dataset']
    #         if 'batch_shapes_cfg' in test_data_cfg:
    #             test_data_cfg.batch_shapes_cfg = None
	
	# build the runner from config

    runner = Runner.from_cfg(cfg)


    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        if cfg.default_scope == 'mmyolo':
            from mmdet.evaluation import DumpDetResults as DumpResults
        if cfg.default_scope in ['mmseg', 'mmcls']:
            from mmengine.evaluator import DumpResults
            cfg.test_evaluator['output_dir'] = args.out
            cfg.test_evaluator['keep_results'] = False
        else: pass 

        runner.test_evaluator.metrics.append(
                DumpResults(out_file_path=args.out))
    
    # start testing
    runner.test()

if __name__ == '__main__':
    main()
