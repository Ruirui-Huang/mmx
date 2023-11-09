from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmx.models.backbones import BaseBackbone
from mmx.models.layers import SPPFBottleneck, CSPRegLayer
from mmx.registry import MODELS


@MODELS.register_module()
class CSPFRegNet(BaseBackbone):
    """CSP-RegNet backbone used in FENet(Free-Efficient Network).

    Args:
        arch (str): Architecture of CSPNeXt, from {P4, P5}.
            Defaults to P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        block_cfg (dict): Config dict for block. Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True)
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', momentum=0.1,
            eps=1e-5).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        attention_cfg (dict): Config dict for `EffectiveSELayer`.
            Defaults to dict(type='EffectiveSELayer',
            act_cfg=dict(type='HSigmoid')).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
        use_large_stem (bool): Whether to use large stem layer.
            Defaults to False.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks
    arch_settings = {
        'P4S': [[64, 64, 2, False], [64, 128, 5, False], [128, 256, 5, True]],
        'P4M': [[64, 96, 3, False], [96, 160, 6, False], [160, 320, 6, True]],
        'P4B': [[64, 128, 3, False], [128, 192, 6, False], [192, 320, 6, True]],
        'P5': [[64, 96, 3, False], [96, 128, 6, False], [128, 160, 6, False], [160, 320, 3, True]]
    }

    def __init__(self,
                 arch: str = 'P4',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 group_width: int = 8,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (1, 2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 arch_ovewrite: dict = None,
                 block_cfg: ConfigType = dict(type='mmx.RepBasicBlock',
                                              shortcut=True,
                                              use_alpha=True),
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 stem_large_kernel_size: bool = False,
                 init_cfg: OptMultiConfig = None):
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        arch_setting = [[
            int(in_channels * widen_factor),
            int(out_channels * widen_factor),
            round(num_blocks * deepen_factor), use_spp
        ] for in_channels, out_channels, num_blocks, use_spp in arch_setting]
        self.block_cfg = block_cfg
        self.arch = arch
        self.group_width = group_width
        self.stem_large_kernel_size = stem_large_kernel_size

        super().__init__(arch_setting,
                         deepen_factor,
                         widen_factor,
                         input_channels=input_channels,
                         out_indices=out_indices,
                         plugins=plugins,
                         frozen_stages=frozen_stages,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         norm_eval=norm_eval,
                         init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        if "P4" in self.arch:
            if self.stem_large_kernel_size:
                stem = nn.Sequential(
                    ConvModule(self.input_channels,
                               self.arch_setting[0][0] // 2,
                               5,
                               stride=4,
                               padding=2,
                               act_cfg=self.act_cfg,
                               norm_cfg=self.norm_cfg),
                    ConvModule(self.arch_setting[0][0] // 2,
                               self.arch_setting[0][0],
                               3,
                               stride=1,
                               padding=1,
                               groups=self.arch_setting[0][0] // self.group_width,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg))
            else:
                stem = nn.Sequential(
                    ConvModule(self.input_channels,
                               self.arch_setting[0][0] // 2,
                               3,
                               stride=2,
                               padding=1,
                               act_cfg=self.act_cfg,
                               norm_cfg=self.norm_cfg),
                    ConvModule(self.arch_setting[0][0] // 2,
                               self.arch_setting[0][0] // 2,
                               3,
                               stride=2,
                               padding=1,
                               groups=self.arch_setting[0][0] // 2 // self.group_width,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(self.arch_setting[0][0] // 2,
                               self.arch_setting[0][0],
                               3,
                               stride=1,
                               padding=1,
                               groups=self.arch_setting[0][0] // self.group_width,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg))
        else:
            stem = nn.Sequential(
                ConvModule(self.input_channels,
                           self.arch_setting[0][0] // 2,
                           3,
                           stride=2,
                           padding=1,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg),
                ConvModule(self.arch_setting[0][0] // 2,
                           self.arch_setting[0][0],
                           3,
                           stride=1,
                           padding=1,
                           groups=self.arch_setting[0][0] // self.group_width,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, use_spp = setting

        # groups = out_channels // self.group_width
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            # groups=groups,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPRegLayer(out_channels,
                                out_channels,
                                group_width=self.group_width,
                                num_blocks=num_blocks,
                                block_cfg=self.block_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(out_channels,
                                 out_channels,
                                 kernel_sizes=5,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()