# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads import PSPHead
from mmseg.models.utils import resize
from mmx.registry import MODELS
import numpy as np
from ..utils import DeconvModule


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
        self.adaptive_layers = nn.ModuleList()
        for pool_scale in pool_scales:
            self.adaptive_layers.append(
                nn.Sequential(
                    AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs),
                    DeconvModule(
                        in_channels=self.channels,
                        out_channels=self.channels,
                        groups=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        scale_factor=int(16/pool_scale),
                        kernel_size=int(16/pool_scale))
                )
            )
            
    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        # for ppm in self:
        #     ppm_out = ppm(x)
        #     upsampled_ppm_out = resize(
        #         ppm_out,
        #         size=x.size()[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        #     ppm_outs.append(upsampled_ppm_out)
        for ppm in self.adaptive_layers:
            ppm_out = ppm(x)
            ppm_outs.append(ppm_out)
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
        
        self.up = DeconvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    groups=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    scale_factor=32,
                    kernel_size=32)
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        if isinstance(self.in_index, (list, tuple)):
            x = self._transform_inputs(inputs[-1])
        else:
            x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        feats = self.up(feats)
        return feats