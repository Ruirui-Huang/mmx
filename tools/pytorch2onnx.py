import argparse, warnings, sys
import os.path as osp
import torch
import onnx
import onnxsim
import numpy as np

from mmengine import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.model import is_model_wrapper
from mmx.models import RepVGGBlock
from merge_bn import fuse_bn_recursively
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MM model to ONNX')
    parser.add_argument('config', help='test config file path', default=None)
    parser.add_argument('checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--caffe', action='store_true', help='Whether to convert caffe.')
    parser.add_argument('--opset-version', type=int, default=12)
    parser.add_argument('--simplify', action='store_true', help='Whether to simplify onnx model.')
    parser.add_argument('--dynamic-export',
                        action='store_true',
                        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    parser.add_argument('--input-names',
                        type=str,
                        nargs='+',
                        default=['data'],
                        help='input layer name')
    parser.add_argument('--output-names',
                        type=str,
                        nargs='+',
                        default=['detect0', 'detect1', 'detect2'],
                        help='output layer name')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')

    args = parser.parse_args()

    return args

def switch_to_deploy(model):
    """Model switch to deploy status."""
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
    print('Switch model to deploy modality.')

def remove_initializer_from_input(input, output):
    model = onnx.load(input)
    if model.ir_version < 4:
        print('Model with ir_version below 4 requires to include initilizer in graph input')
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, output)

def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps, module.momentum,
                                             module.affine, module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output

def pytorch2onnx(
        model,
        input_shape,
        input_names=['data'],
        output_names=['segout'],
        opset_version=11,
        dynamic_export=False,
        show=False,
        output_file='model.onnx',
    forward_particular=False,
        do_simplify=False):
    """Export Pytorch model to ONNX model

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.

    """
    model.cpu().eval()

    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)

    imgs = torch.FloatTensor(imgs).requires_grad_(True)

    input_img = imgs
    if hasattr(model, 'forward_onnx'):
        model.forward = model.forward_onnx
    elif hasattr(model, 'forward_particular') and forward_particular:
        model.forward = model.forward_particular
    elif hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        model.forward = model.forward

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {'input': {0: 'batch', 2: 'width', 3: 'height'}, 'probs': {0: 'batch'}}
    else:
        dynamic_axes = {}

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_img,
            output_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)

    if dynamic_axes:
        input_shape = (input_shape[0], input_shape[1], input_shape[2] * 2, input_shape[3] * 2)
    else:
        input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    input_dic = {'data': imgs.detach().cpu().numpy()}
    input_shape_dic = {'data': list(input_shape)}

    model_opt, check_ok = onnxsim.simplify(
        output_file,
        overwrite_input_shapes=input_shape_dic,
        input_data=input_dic,
        dynamic_input_shape=dynamic_export)
    
    if check_ok:
        onnx.save(model_opt, output_file)
        remove_initializer_from_input(output_file, output_file)
        print(f'Successfully exported and simplified ONNX model: {output_file}')
    else:
        warnings.warn('Failed to simplify ONNX model.')

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # 减少不必要的日志打印
    cfg.log_level = 'ERROR'
    # cfg.load_from = args.checkpoint
    input_size = cfg.get('img_scale', None)
    # 判断当前任务的前向方式
    forward_particular = cfg.get('forward_particular', None)
    print(f'The current network input size is {input_size}')
    if input_size:
        input_shape = (1, 3, input_size[0], input_size[1])
    else:
        raise ValueError('Not find [input_size] of dataset in config, please check!!!')

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    model = _convert_batchnorm(model)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    switch_to_deploy(model)
    model = fuse_bn_recursively(model)
       
    default_scope = cfg.default_scope
    if "cls" in default_scope:
        output_names = ['prob']
    elif "seg" in default_scope:
        output_names = ['prob']
    elif "det" in default_scope or "yolo" in default_scope:
        output_names = args.output_names
    output_file = osp.join(osp.dirname(args.checkpoint), 'model.onnx')
    pytorch2onnx(
        model,
        input_shape,
        input_names=args.input_names,
        output_names=output_names,
        opset_version=args.opset_version,
        show=args.show,
        dynamic_export=args.dynamic_export,
        output_file=output_file,
        forward_particular=forward_particular,
        do_simplify=args.simplify)
