import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.model import BaseModule
# from torch.nn.modules.batchnorm import _BatchNorm

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


class DeconvUpBranch(BaseModule):

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=[128, 128, 128, 128],
                 multicat=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(DeconvUpBranch, self).__init__(init_cfg)
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
                         scale_factor=2,
                         kernel_size=2,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg))
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
                         scale_factor=2,
                         kernel_size=2,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg))
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
                         scale_factor=2,
                         kernel_size=2,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg))
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
                                        scale_factor=2,
                                        kernel_size=2,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)
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
            self.fam_32_upcat = nn.Sequential(
                DeconvModule(in_channels=out_channels[3],
                             out_channels=out_channels[3] // 2,
                             scale_factor=2,
                             kernel_size=2,
                             norm_cfg=norm_cfg,
                             act_cfg=act_cfg),
                DeconvModule(in_channels=out_channels[3] // 2,
                             out_channels=out_channels[3] // 4,
                             scale_factor=2,
                             kernel_size=2,
                             norm_cfg=norm_cfg,
                             act_cfg=act_cfg),
                DeconvModule(in_channels=out_channels[3] // 4,
                             out_channels=out_channels[3] // 8,
                             scale_factor=2,
                             kernel_size=2,
                             norm_cfg=None,
                             act_cfg=None),
                ConvModule(out_channels[3] // 8, out_channels[3], kernel_size=1, norm_cfg=norm_cfg),
                ConvModule(out_channels[3],
                           out_channels[3],
                           kernel_size=3,
                           padding=1,
                           norm_cfg=norm_cfg))
            self.fam_16_upcat = nn.Sequential(
                DeconvModule(in_channels=out_channels[2],
                             out_channels=out_channels[2] // 2,
                             scale_factor=2,
                             kernel_size=2,
                             norm_cfg=norm_cfg,
                             act_cfg=act_cfg),
                DeconvModule(in_channels=out_channels[2] // 2,
                             out_channels=out_channels[2] // 4,
                             scale_factor=2,
                             kernel_size=2,
                             norm_cfg=None,
                             act_cfg=None),
                ConvModule(out_channels[2] // 4, out_channels[2], kernel_size=1, norm_cfg=norm_cfg),
                ConvModule(out_channels[2],
                           out_channels[2],
                           kernel_size=3,
                           padding=1,
                           norm_cfg=norm_cfg))

            self.out_feature_channels = sum(out_channels)
        else:
            self.out_feature_channels = out_channels[0] + out_channels[1]

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
            out = torch.cat([smfeat_4, upfeat_8to4, upfeat_16to4, upfeat_32to4], dim=1)
        else:
            out = torch.cat((smfeat_4, upfeat_8to4), dim=1)

        return out


@MODELS.register_module()
class USemHead(BaseDecodeHead):
    """
    Unified Semantic segmentation Head
    """

    def __init__(self,
                 head_type="S",
                 base_chans=[64, 128, 256, 512],
                 scale_factor=4,
                 with_bn=False,
                 use_adapter_conv=True,
                 multicat=False,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        if self.input_transform == 'resize_concat':
            if isinstance(self.in_index, (list, tuple)):
                self.upsamples = nn.ModuleList()
                for i in self.in_index:
                    channel = kwargs['in_channels'][i]
                    self.upsamples.append(
                        DeconvModule(in_channels=channel,
                                     out_channels=channel,
                                     groups=1,
                                     norm_cfg=self.norm_cfg,
                                     act_cfg=self.act_cfg,
                                     scale_factor=2**i,
                                     kernel_size=2**i))
        if with_bn:
            deconv = nn.ConvTranspose2d(self.channels,
                                        self.num_classes,
                                        kernel_size=scale_factor,
                                        stride=scale_factor)
            norm_name, norm = build_norm_layer(self.norm_cfg, self.num_classes)
            self.conv_seg = nn.Sequential(deconv, norm)

        else:
            self.conv_seg = nn.ConvTranspose2d(self.channels,
                                               self.num_classes,
                                               kernel_size=scale_factor,
                                               stride=scale_factor)

        layers = []
        if head_type.startswith("L"):
            base_chans = [64, 128, 256, 512]
        elif head_type.startswith("M"):
            base_chans = [64, 128, 128, 256]
        elif head_type.startswith("S"):
            base_chans = [128, 128, 128, 128]

        if use_adapter_conv:
            layers.append(AdapterConv(self.in_channels, base_chans))
            in_channels = base_chans[:]
        else:
            in_channels = self.in_channels[:]

        if head_type == "L":
            layer = DeconvUpBranch(in_channels, [128, 128, 128, 128], multicat=multicat)
            # layers.append(DeconvUpBranch(in_channels, multicat=multicat))
        elif head_type == "M":
            layer = DeconvUpBranch(in_channels, [96, 96, 64, 32], multicat=multicat)
            # layers.append(DeconvUpBranch(in_channels, [96, 96, 64, 32], multicat=multicat))
        elif head_type == "S":
            layer = DeconvUpBranch(in_channels, [128, 32, 16, 16], multicat=multicat)
            # layers.append(DeconvUpBranch(in_channels, [128, 32, 16, 16], multicat=multicat))
        else:
            raise ValueError(f"Unknown USemHead type {head_type}")

        layers.append(layer)
        self.layers = nn.Sequential(*layers)

        self.bottleneck = ConvModule(layer.out_feature_channels,
                                     self.channels,
                                     3,
                                     padding=1,
                                     conv_cfg=self.conv_cfg,
                                     norm_cfg=self.norm_cfg,
                                     act_cfg=self.act_cfg)

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [self.upsamples[i](x) for i, x in enumerate(inputs)]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = self.layers(inputs)
        feats = self.bottleneck(outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output