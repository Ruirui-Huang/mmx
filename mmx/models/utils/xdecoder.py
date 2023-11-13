import torch
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from .utils import DeconvModule, SEModule, ReparamLargeKernelConv, ReparamAsymKernelConv

class XDecoder(BaseModule):

    def __init__(self,
                 channels,
                 expansion=1,
                 use_se=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(XDecoder, self).__init__(init_cfg=init_cfg)
        self.use_se = use_se
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvModule(in_channels=channels16,
                                 out_channels=128 * expansion,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg)
        self.head16up = DeconvModule(in_channels=128 * expansion,
                                     out_channels=128 * expansion,
                                     groups=1,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg,
                                     scale_factor=2,
                                     kernel_size=2)
        self.head8 = ConvModule(in_channels=channels8,
                                out_channels=128 * expansion,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.head4 = ConvModule(in_channels=channels4,
                                out_channels=16 * expansion,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        # self.conv8 = ConvModule(in_channels=128 * expansion,
        #                         out_channels=64 * expansion,
        #                         kernel_size=3,
        #                         stride=1,
        #                         padding=1,
        #                         norm_cfg=norm_cfg,
        #                         act_cfg=act_cfg)
        # self.conv8 = ReparamLargeKernelConv(in_channels=128 * expansion,
        #                                     out_channels=64 * expansion,
        #                                     kernel_size=5,
        #                                     stride=1,
        #                                     groups=8,
        #                                     dilation=1,
        #                                     small_kernel=3,
        #                                     deploy=False,
        #                                     norm_cfg=norm_cfg,
        #                                     act_cfg=act_cfg)
        self.conv8 = ReparamAsymKernelConv(in_channels=128 * expansion,
                                           out_channels=64 * expansion,
                                           kernel_size=3,
                                           stride=1,
                                           groups=8,
                                           dilation=1,
                                           deploy=False,
                                           norm_cfg=norm_cfg,
                                           act_cfg=act_cfg)
        if self.use_se:
            self.se = SEModule(64 * expansion, 64 * expansion // 4)
        self.head8up = DeconvModule(in_channels=64 * expansion,
                                    out_channels=64 * expansion,
                                    groups=1,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    scale_factor=2,
                                    kernel_size=2)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]

        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = self.head16up(x16)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        if self.use_se:
            x8 = self.se(x8)
        x8 = self.head8up(x8)
        x4 = torch.cat((x8, x4), dim=1)
        # outs = [x4, x["8"]]
        outs = [x4, x["8"], x["4"]]
        return tuple(outs)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()