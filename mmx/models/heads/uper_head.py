import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.utils import resize
from mmx.registry import MODELS
from mmseg.models.decode_heads import UPerHead
from mmx.models.heads.psp_head import PPM
from ..utils import DeconvModule

@MODELS.register_module()
class DetUPerHead(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(DetUPerHead, self).__init__(**kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        
        self.top_down_path = nn.ConvTranspose2d(
            self.channels,
            self.channels,                                
            kernel_size=2,
            stride=2)
        
        self.up_heads = nn.ModuleList()
        for i in range(len(self.in_channels)-1):
            self.up_heads.append(
                nn.ConvTranspose2d(
                    self.channels,
                    self.channels,                                
                    kernel_size=int(2**(i+1)),
                    stride=int(2**(i+1)))
            )

        self.up = DeconvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            groups=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            scale_factor=4,
            kernel_size=4)
    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            laterals[i-1] = laterals[i-1] + self.top_down_path(laterals[i])

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            # fpn_outs[i] = resize(
            #     fpn_outs[i],
            #     size=fpn_outs[0].shape[2:],
            #     mode='bilinear',
            #     align_corners=self.align_corners)
        
            fpn_outs[i] = self.up_heads[i-1](fpn_outs[i])
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        feats = self.up(feats)
        return feats