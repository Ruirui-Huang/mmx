# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, ConvModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmdet.models.backbones.csp_darknet import CSPLayer
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.utils import make_divisible, make_round
from mmx.registry import MODELS


@MODELS.register_module()
class VitDetPAFPN(BaseModule):
    """Path Aggregation Network used in VitDet.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dim,
                 out_channels,
                 rescales=[4, 2, 1, 0.5],
                 widen_factor: float = 1.0,
                 in_index=-1,
                 input_transform=None,
                 upsample_feats_cat_first: bool = True,
                 num_csp_blocks: int = 1,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = num_csp_blocks
        super().__init__(init_cfg)

        # self.in_channels = in_channels
        self.input_transform = input_transform
        self.rescales = rescales
        self.in_index = in_index
        self.out_channels = make_divisible(out_channels, widen_factor)
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.sfp_layers = []
        for idx, scale in enumerate(self.rescales):
            out_dim = embed_dim
            if scale == 4:
                layers = [
                    nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim // 2)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = embed_dim // 4
            elif scale == 2:
                layers = [nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)]
                out_dim = embed_dim // 2
            elif scale == 1:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif scale == 0.25:
                layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
            else:
                raise KeyError(f'invalid {scale} for SimpleFeaturePyramid')
            layers.extend([
                ConvModule(out_dim, self.out_channels, kernel_size=1, norm_cfg=norm_cfg),
                ConvModule(self.out_channels,
                           self.out_channels,
                           kernel_size=3,
                           padding=1,
                           norm_cfg=norm_cfg)
            ])
            layers = nn.Sequential(*layers)
            self.add_module(f"sfp_{idx}", layers)
            self.sfp_layers.append(layers)

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(rescales) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(rescales) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        self.out_layers = nn.ModuleList()
        for idx in range(len(rescales)):
            self.out_layers.append(self.build_out_layer(idx))

    def init_weights(self):
        if self.init_cfg is None:
            """Initialize the parameters."""
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int):
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayer(self.out_channels * 2,
                        self.out_channels,
                        num_blocks=self.num_csp_blocks,
                        add_identity=False,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(self.out_channels,
                          self.out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayer(self.out_channels * 2,
                        self.out_channels,
                        num_blocks=self.num_csp_blocks,
                        add_identity=False,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

            Args:
                inputs (list[Tensor]): List of multi-level img features.

            Returns:
                Tensor: The transformed inputs
            """
        if self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = [inputs[self.in_index] for i in range(len(self.rescales))]

        return inputs

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        assert len(inputs) == len(self.rescales)
        # reduce layers
        # reduce_outs = []
        # for idx in range(len(self.rescales)):
        #     reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # sfp_layers
        sfp_outs = []
        for idx in range(len(self.rescales)):
            sfp_layer = getattr(self, f'sfp_{idx}')
            sfp_outs.append(sfp_layer(inputs[idx]))

        # top-down path
        inner_outs = [sfp_outs[-1]]
        for idx in range(len(self.rescales) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = sfp_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.rescales) - 1 - idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.rescales) - 1 - idx](top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.rescales) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.rescales)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
