import torch
import torch.nn as nn
from mmengine.model import constant_init, kaiming_init
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor
from mmcv.cnn import ConvModule, build_norm_layer, build_activation_layer
# from .upsample import DeconvModule, Upsample
from typing import Dict, List, Optional, Union
from mmx.registry import MODELS
from mmseg.utils import OptConfigType
import torch.nn.functional as F

def Upsample(in_channels, out_channels, scale_factor):
    stride = scale_factor
    ks = scale_factor
    pad = 0
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=pad)


class DeconvModule(BaseModule):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 kernel_size=4,
                 scale_factor=2,
                 init_cfg=None):
        super(DeconvModule, self).__init__(init_cfg)

        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        bias = not self.with_norm
        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        deconv = nn.ConvTranspose2d(in_channels,
                                    out_channels,
                                    groups=groups,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)

        if self.with_norm:
            norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        else:
            norm = nn.Identity()
        if self.with_activation:
            activate = build_activation_layer(act_cfg)
        else:
            activate = nn.Identity()
        self.upsamping = nn.Sequential(deconv, norm, activate)
        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                    nonlinearity = 'leaky_relu'
                    a = self.act_cfg.get('negative_slope', 0.01)
                else:
                    nonlinearity = 'relu'
                    a = 0
                kaiming_init(m, a=a, nonlinearity=nonlinearity)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1, bias=0)

    def forward(self, x):
        """Forward function."""
        x = self.upsamping(x)

        return x


class SEModule(BaseModule):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self,
                 w_in,
                 w_se,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels=w_in,
                                out_channels=w_se,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                conv_cfg=conv_cfg,
                                act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=w_se,
                                out_channels=w_in,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                conv_cfg=conv_cfg,
                                act_cfg=act_cfg[1])

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.conv2(y)
        return x * y


class ReparamAsymKernelConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups,
                 dilation=1,
                 deploy=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(ReparamAsymKernelConv, self).__init__()
        self.deploy = deploy
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        padding = dilation * (kernel_size // 2)
        if deploy:
            self.fused_conv = ConvModule(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         norm_cfg=None,
                                         act_cfg=None)
        else:
            self.square_conv = ConvModule(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(kernel_size, kernel_size),
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          norm_cfg=norm_cfg,
                                          act_cfg=None)

            if padding - dilation * kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = ConvModule(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(kernel_size, 1),
                                       stride=stride,
                                       padding=ver_padding,
                                       dilation=dilation,
                                       groups=groups,
                                       norm_cfg=norm_cfg,
                                       act_cfg=None)

            self.hor_conv = ConvModule(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, kernel_size),
                                       stride=stride,
                                       padding=hor_padding,
                                       dilation=dilation,
                                       groups=groups,
                                       norm_cfg=norm_cfg,
                                       act_cfg=None)
            self.activate = build_activation_layer(act_cfg)

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2:square_h // 2 - asym_h // 2 + asym_h,
                      square_w // 2 - asym_w // 2:square_w // 2 - asym_w // 2 +
                      asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv.conv, self.hor_conv.norm)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv.conv, self.ver_conv.norm)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv.conv, self.square_conv.norm)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels,
                                    out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size,
                                    stride=self.square_conv.stride,
                                    padding=self.square_conv.padding,
                                    dilation=self.square_conv.dilation,
                                    groups=self.square_conv.groups)
        self.fused_conv.weight.data = eq_k
        self.fused_conv.bias.data = eq_b
        self.__delattr__('square_conv')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')

    def forward(self, input):
        if hasattr(self, "fused_conv"):
            result = self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            horizontal_outputs = self.hor_conv(hor_input)
            result = square_outputs + vertical_outputs + horizontal_outputs

        return self.activate(result)
        # return result

class ReparamLargeKernelConv(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups,
                 small_kernel,
                 dilation=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 deploy=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        padding = dilation * (kernel_size // 2)
        if deploy:
            self.lkb_reparam = ConvModule(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          norm_cfg=None,
                                          act_cfg=None)
        else:
            self.lkb_origin = ConvModule(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         norm_cfg=norm_cfg,
                                         act_cfg=None)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = ConvModule(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=small_kernel,
                                             stride=stride,
                                             padding=small_kernel // 2,
                                             groups=groups,
                                             dilation=1,
                                             norm_cfg=norm_cfg,
                                             act_cfg=None)

        self.activate = build_activation_layer(act_cfg)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return self.activate(out)
        # return out

    def fuse_bn(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = self.fuse_bn(self.lkb_origin.conv, self.lkb_origin.norm)
        if hasattr(self, 'small_conv'):
            small_k, small_b = self.fuse_bn(self.small_conv.conv, self.small_conv.norm)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size,
                                     stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding,
                                     dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class BasicBlock(BaseModule):
    """Basic block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at the
            last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.downsample = downsample
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out

class Bottleneck(BaseModule):
    """Bottleneck block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at
            the last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 2

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            3,
            stride,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            channels,
            channels * self.expansion,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out
        
        
class DAPPM(BaseModule):
    """DAPPM module in `DDRNet <https://arxiv.org/abs/2101.06085>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.upsamples = nn.ModuleList()
        for i in range(1, self.num_scales):
            self.upsamples.append(
                DeconvModule(
                    in_channels=self.branch_channels,
                    out_channels=self.branch_channels,
                    groups=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    scale_factor=min(2**i, 8),
                    kernel_size=min(2**i, 8)
                )
            )

    def forward(self, inputs):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = self.upsamples[i-1](self.scales[i](inputs))
            # feat_up = F.interpolate(
            #     self.scales[i](inputs),
            #     size=inputs.shape[2:],
            #     mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)
