import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.model import BaseModule
# from torch.nn.modules.batchnorm import _BatchNorm

from mmx.registry import MODELS


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
                 align_corners=False,
                 init_cfg=None):
        super(UpBranch, self).__init__(init_cfg)

        self.align_corners = align_corners
        self.multicat = multicat  # 是否多层拼接操作segout，提高精度

        self.fam_32_up = ConvModule(in_channels[3],
                                    in_channels[2],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.fam_16_up = ConvModule(in_channels[2],
                                    in_channels[1],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.fam_8_sm = ConvModule(in_channels[1],
                                   out_channels[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg)
        self.fam_8_up = ConvModule(in_channels[1],
                                   in_channels[0],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg)
        self.fam_4 = ConvModule(in_channels[0],
                                out_channels[0],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                conv_cfg=conv_cfg,
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

            self.out_feature_channels = sum(out_channels)
        else:
            self.out_feature_channels = out_channels[0] + out_channels[1]

    def forward(self, x):

        feat4, feat8, feat16, feat32 = x

        upfeat_32 = self.fam_32_up(feat32)

        _, _, H, W = feat16.size()
        x = F.interpolate(upfeat_32,
                          (H, W), mode='bilinear', align_corners=self.align_corners) + feat16
        upfeat_16 = self.fam_16_up(x)
        record = []
        if self.multicat:
            smfeat_32 = self.fam_32_sm(feat32)
            smfeat_16 = self.fam_16_sm(x)
            record = [smfeat_32, smfeat_16]

        _, _, H, W = feat8.size()
        x = F.interpolate(upfeat_16,
                          (H, W), mode='bilinear', align_corners=self.align_corners) + feat8
        smfeat_8 = self.fam_8_sm(x)
        record.append(smfeat_8)
        upfeat_8 = self.fam_8_up(x)

        _, _, H, W = feat4.size()
        smfeat_4 = self.fam_4(
            F.interpolate(upfeat_8, (H, W), mode='bilinear', align_corners=self.align_corners) +
            feat4)
        record.append(smfeat_4)
        out = smfeat_4
        for i in range(len(record) - 2, -1, -1):
            out = torch.cat([
                out,
                F.interpolate(record[i], (H, W), mode='bilinear', align_corners=self.align_corners)
            ],
                            dim=1)

        return out


@MODELS.register_module()
class PSemHead(BaseDecodeHead):
    """
    Particular Semantic segmentation Head
    """

    def __init__(self,
                 head_type="S",
                 base_chans=[64, 128, 256, 512],
                 use_adapter_conv=True,
                 multicat=False,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

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
            layer = UpBranch(in_channels, [128, 128, 128, 128],
                             multicat=multicat,
                             align_corners=self.align_corners)
        elif head_type == "M":
            layer = UpBranch(in_channels, [96, 96, 64, 32],
                             multicat=multicat,
                             align_corners=self.align_corners)
        elif head_type == "S":
            layer = UpBranch(in_channels, [128, 32, 16, 16],
                             multicat=multicat,
                             align_corners=self.align_corners)
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
