import torch
import torch.nn as nn

from mmcv.cnn import build_norm_layer
from mmseg.models.decode_heads import FCNHead
from mmx.registry import MODELS
from ..ops import DeconvModule


@MODELS.register_module()
class SegFCNHead(FCNHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self, scale_factor=4, with_bn=False, **kwargs):
        super(SegFCNHead, self).__init__(**kwargs)

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
