# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer, ConvModule
from mmengine.model import BaseModule

from mmx.registry import MODELS


@MODELS.register_module()
class SimpleFeaturePyramid(BaseModule):
    """SimpleFeaturePyramid.
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dim,
                 out_channels,
                 in_index=-1,
                 rescales=[4, 2, 1, 0.5],
                 input_transform=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.rescales = rescales
        self.in_index = in_index
        self.input_transform = input_transform
        # self.upsample_4x = None
        self.simfp_stages = []
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
                ConvModule(out_dim, out_channels, kernel_size=1, norm_cfg=norm_cfg),
                ConvModule(out_channels, out_channels, kernel_size=3, padding=1, norm_cfg=norm_cfg)
            ])
            layers = nn.Sequential(*layers)
            self.add_module(f"simfp_{idx}", layers)
            self.simfp_stages.append(layers)

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
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        outputs = []
        # for stage in self.simfp_stages:
        #     outputs.append(stage(x))
        for i in range(len(self.simfp_stages)):
            stage = getattr(self, f'simfp_{i}')
            outputs.append(stage(x))
        return tuple(outputs)
