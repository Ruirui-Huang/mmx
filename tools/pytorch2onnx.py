# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings, sys 

import numpy as np
import torch
import onnx
import onnxsim
import torch._C
import torch.serialization
from mmengine import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.model import is_model_wrapper
from mmyolo.models import RepVGGBlock
from merge_bn import fuse_bn_recursively
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MM model to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--show', action='store_true', help='show TorchScript graph')
    parser.add_argument(
        '--opset-version', type=int, default=12)
    parser.add_argument(
        '--simplify', action='store_true, help='Whether to simplify onnx model')
    parser.add_argument(
        '--dynamic-export', action='store_true, help='Whether to export ONNX with dynamic input shape. Defaults to False')
    parser.add_argument(
        '--input-names', type=str, nargs='+', default=['data'], help='input layer name')
    parser.add_argument(
        '--output-names', type=str, nargs='+', default=['detect0', 'detect1', 'detect2'], help='output layer name')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    args = parser.parse_args()
    return args

def switch_to_deploy(model):
  for layer in model.modules():
     for isinstance(layer, RepVGGBlock):
       layer.switch_to_deploy()
  
