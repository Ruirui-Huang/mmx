# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads import PSPHead
from mmseg.models.utils import resize
from mmx.registry import MODELS
import numpy as np


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        if isinstance(output_size, int):
            self.output_size = np.array([output_size, output_size])
        elif isinstance(output_size, list):
            self.output_size = np.array(output_size)
        else:
            print("check!")

    def forward(self, x):
        shape_x = x.shape
        if (shape_x[-1] < self.output_size[-1]):
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] -shape_x[-1]))
            paddzero = paddzero.to('cuda:0')
            x = torch.cat((x, paddzero), axis=-1)
        
        stride_size = np.floor(np.array(x.shape[-2:])/self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1)*stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x 
    
class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)),
                    
            )
            
    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@MODELS.register_module()
class DetPSPHead(PSPHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)