import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmx.registry import MODELS

from ..utils import DeconvModule


class AdapterConv(BaseModule):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=[64, 128, 256, 512],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(AdapterConv, self).__init__()
        assert len(in_channels) == len(
            out_channels), "Number of input and output branches should match"
        self.adapter_conv = nn.ModuleList()

        for k in range(len(in_channels)):
            self.adapter_conv.append(
                ConvModule(in_channels[k],
                           out_channels[k],
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

    def forward(self, x):
        out = []
        for k in range(len(self.adapter_conv)):
            out.append(self.adapter_conv[k](x[k]))
        return tuple(out)


class UpBranch(BaseModule):

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=[128, 128, 128, 128],
                 multicat=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(UpBranch, self).__init__(init_cfg)
        self.multicat = multicat  # 是否多层拼接操作segout，提高精度

        self.fam_32_up = nn.Sequential(
            ConvModule(in_channels[3],
                       in_channels[2],
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            DeconvModule(in_channels=in_channels[2],
                         out_channels=in_channels[2],
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         scale_factor=2,
                         kernel_size=2))
        self.fam_16_up = nn.Sequential(
            ConvModule(in_channels[2],
                       in_channels[1],
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            DeconvModule(in_channels=in_channels[2],
                         out_channels=in_channels[2],
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         scale_factor=2,
                         kernel_size=2))
        self.fam_8_sm = ConvModule(in_channels[1],
                                   out_channels[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg)
        self.fam_8_up = nn.Sequential(
            ConvModule(in_channels[1],
                       in_channels[0],
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            DeconvModule(in_channels=in_channels[0],
                         out_channels=in_channels[0],
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         scale_factor=2,
                         kernel_size=2))
        self.fam_4 = ConvModule(in_channels[0],
                                out_channels[0],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.fam_8_upcat = DeconvModule(in_channels=out_channels[1],
                                        out_channels=out_channels[1],
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg,
                                        scale_factor=2,
                                        kernel_size=2)
        if self.multicat:
            self.fam_32_sm = ConvModule(in_channels[3],
                                        out_channels[3],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)
            self.fam_16_sm = ConvModule(in_channels[2],
                                        out_channels[2],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)
            self.fam_32_upcat = DeconvModule(in_channels=out_channels[3],
                                             out_channels=out_channels[3],
                                             norm_cfg=norm_cfg,
                                             act_cfg=act_cfg,
                                             scale_factor=8,
                                             kernel_size=8)
            self.fam_16_upcat = DeconvModule(in_channels=out_channels[2],
                                             out_channels=out_channels[2],
                                             norm_cfg=norm_cfg,
                                             act_cfg=act_cfg,
                                             scale_factor=4,
                                             kernel_size=4)

    def forward(self, x):

        feat4, feat8, feat16, feat32 = x

        upfeat_32 = self.fam_32_up(feat32)

        x = upfeat_32 + feat16
        if self.multicat:
            smfeat_32 = self.fam_32_sm(feat32)
            smfeat_16 = self.fam_16_sm(x)
        upfeat_16 = self.fam_16_up(x)

        x = upfeat_16 + feat8
        smfeat_8 = self.fam_8_sm(x)
        upfeat_8 = self.fam_8_up(x)

        smfeat_4 = self.fam_4(upfeat_8 + feat4)

        upfeat_8to4 = self.fam_8_upcat(smfeat_8)
        if self.multicat:
            upfeat_16to4 = self.fam_16_upcat(smfeat_16)
            upfeat_32to4 = self.fam_32_upcat(smfeat_32)
            feat_upcat_4 = torch.cat([smfeat_4, upfeat_8to4, upfeat_16to4, upfeat_32to4], dim=1)
            out = [smfeat_32, smfeat_16, smfeat_8, feat_upcat_4]
        else:
            feat_upcat_4 = torch.cat((smfeat_4, upfeat_8to4), dim=1)
            out = [upfeat_32, upfeat_16, smfeat_8, feat_upcat_4]

        return tuple(out)


@MODELS.register_module()
class FEINetUpHead(BaseModule):
    arch_settings = {
        'P4S': [[64, 64, 128, 256]],
        'P4M': [[64, 96, 160, 320]],
        'P4B': [[64, 128, 192, 320]],
        'P5': [[64, 128, 256, 512]]
    }

    def __init__(self,
                 arch="P4S",
                 widen_factor=1,
                 head_type="S",
                 base_chans=[64, 128, 256, 512],
                 freeze_all=False,
                 use_adapter_conv=True,
                 multicat=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(FEINetUpHead, self).__init__(init_cfg)
        self.freeze_all = freeze_all
        arch_setting = self.arch_settings[arch]
        in_channels = [int(in_channel * widen_factor) for in_channel in arch_setting[0]]
        layers = []
        if head_type.startswith("L"):
            base_chans = [64, 128, 256, 512]
        elif head_type.startswith("M"):
            base_chans = [64, 128, 128, 256]
        elif head_type.startswith("S"):
            base_chans = [128, 128, 128, 128]

        if use_adapter_conv:
            layers.append(AdapterConv(in_channels, base_chans, norm_cfg=norm_cfg, act_cfg=act_cfg))
            in_channels = base_chans[:]

        if head_type == "L":
            layers.append(UpBranch(in_channels, multicat=multicat, norm_cfg=norm_cfg, act_cfg=act_cfg))
        elif head_type == "M":
            layers.append(UpBranch(in_channels, [96, 96, 64, 32], multicat=multicat, norm_cfg=norm_cfg, act_cfg=act_cfg))
        elif head_type == "S":
            layers.append(UpBranch(in_channels, [128, 32, 16, 16], multicat=multicat, norm_cfg=norm_cfg, act_cfg=act_cfg))
        else:
            raise ValueError(f"Unknown FEINetUpHead type {head_type}")

        self.layers = nn.Sequential(*layers)
        self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        if self.freeze_all:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        self._freeze_all()

    def forward(self, x):
        return self.layers(x)


@MODELS.register_module()
class FEINetUpHeadv2(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 256, 512]],
        'P9t': [[32, 64, 96, 128]]
    }

    def __init__(self,
                 arch="P4S",
                 widen_factor=1,
                 head_type="S",
                 base_chans=[64, 128, 256, 512],
                 freeze_all=False,
                 use_adapter_conv=True,
                 multicat=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(FEINetUpHeadv2, self).__init__(init_cfg)
        self.freeze_all = freeze_all
        arch_setting = self.arch_settings[arch]
        in_channels = [int(in_channel * widen_factor) for in_channel in arch_setting[0]]
        layers = []
        if head_type.startswith("L"):
            base_chans = [64, 128, 256, 512]
        elif head_type.startswith("M"):
            base_chans = [64, 128, 128, 256]
        elif head_type.startswith("S"):
            base_chans = [128, 128, 128, 128]

        if use_adapter_conv:
            layers.append(AdapterConv(in_channels, base_chans, norm_cfg=norm_cfg, act_cfg=act_cfg))
            in_channels = base_chans[:]

        if head_type == "L":
            layers.append(UpBranch(in_channels, multicat=multicat, norm_cfg=norm_cfg, act_cfg=act_cfg))
        elif head_type == "M":
            layers.append(UpBranch(in_channels, [96, 96, 64, 32], multicat=multicat, norm_cfg=norm_cfg, act_cfg=act_cfg))
        elif head_type == "S":
            layers.append(UpBranch(in_channels, [128, 32, 16, 16], multicat=multicat, norm_cfg=norm_cfg, act_cfg=act_cfg))
        else:
            raise ValueError(f"Unknown FEINetUpHead type {head_type}")

        self.layers = nn.Sequential(*layers)
        self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        if self.freeze_all:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        self._freeze_all()

    def forward(self, x):
        return self.layers(x)
