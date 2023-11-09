from typing import Optional, Sequence, Tuple, Union
import warnings
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor
from mmyolo.models.layers import RepVGGBlock

from mmx.registry import MODELS


class SPPFBottleneck(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer for
    YOLOv5, YOLOX and PPYOLOE by Glenn Jocher

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(in_channels,
                                    mid_channels,
                                    1,
                                    stride=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(kernel_size=kernel_sizes,
                                         stride=1,
                                         padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(conv2_in_channels,
                                out_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPRegLayer(BaseModule):
    """PPYOLOE Backbone Stage.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_block (int): Number of blocks in this stage.
        block_cfg (dict): Config dict for block. Default config is
            suitable for PPYOLOE+ backbone. And in PPYOLOE neck,
            block_cfg is set to dict(type='RepBasicBlock',
            shortcut=False, use_alpha=False). Defaults to
            dict(type='RepBasicBlock', shortcut=True, use_alpha=True).
        stride (int): Stride of the convolution. In backbone, the stride
            must be set to 2. In neck, the stride must be set to 1.
            Defaults to 1.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        attention_cfg (dict, optional): Config dict for `EffectiveSELayer`.
            Defaults to dict(type='EffectiveSELayer',
            act_cfg=dict(type='HSigmoid')).
        use_spp (bool): Whether to use `SPPFBottleneck` layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 group_width: int = 1,
                 num_blocks: int = 1,
                 block_cfg: ConfigType = dict(type='RepBasicBlock', shortcut=True, use_alpha=True),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        self.num_blocks = num_blocks
        self.block_cfg = block_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.mid_channels = int(out_channels * expand_ratio)
        self.group_width = group_width
        self.main_conv = ConvModule(in_channels,
                                    2 * self.mid_channels,
                                    1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.blocks = self.build_blocks_layer(self.mid_channels)
        self.final_conv = ConvModule((2 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

    def build_blocks_layer(self, blocks_channels: int) -> nn.Module:
        """Build blocks layer.

        Args:
            blocks_channels: The channels of this Module.
        """
        # blocks = nn.ModuleList()
        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(in_channels=blocks_channels,
                 out_channels=blocks_channels,
                 group_width=self.group_width))
        block_cfg.setdefault('norm_cfg', self.norm_cfg)
        block_cfg.setdefault('act_cfg', self.act_cfg)

        # for i in range(self.num_blocks):
        #     # blocks.add_module(str(i), MODELS.build(block_cfg))
        #     blocks.append(MODELS.build(block_cfg))
        blocks = nn.ModuleList(MODELS.build(block_cfg) for _ in range(self.num_blocks))

        return blocks

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        y = self.final_conv(torch.cat(x_main, 1))
        return y


@MODELS.register_module()
class RepBasicBlock(BaseModule):
    """Rep Backbone BasicBlock.

    Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         shortcut (bool): Whether to add inputs and outputs together
         at the end of this layer. Defaults to True.
         use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 group_width: int,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 shortcut: bool = True,
                 use_alpha: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        assert act_cfg is None or isinstance(act_cfg, dict)
        if out_channels % group_width:
            warnings.warn('group_width is invalid, groups will be set 1.')
            groups = 1
        else:
            groups = out_channels // group_width
        self.conv1 = ConvModule(in_channels,
                                out_channels,
                                3,
                                groups=groups,
                                stride=1,
                                padding=1,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)

        self.conv2 = RepVGGBlock(out_channels,
                                 out_channels,
                                 groups=groups,
                                 use_alpha=use_alpha,
                                 act_cfg=act_cfg,
                                 norm_cfg=norm_cfg,
                                 use_bn_first=False)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class Attention(BaseModule):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 proj_drop=0.,
                 downsample_rate: int = 1,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs) -> None:
        super().__init__(init_cfg)
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None):
        # Input projections
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Separate into heads
        query = self._separate_heads(query, self.num_heads)
        key = self._separate_heads(key, self.num_heads)
        value = self._separate_heads(value, self.num_heads)

        # Attention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        out = identity + self.dropout_layer(self.proj_drop(out))

        return out


class AttentionBlock(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = Attention(embed_dims, num_heads=num_heads, proj_drop=drop_rate)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(embed_dims=embed_dims,
                 feedforward_channels=feedforward_channels,
                 num_fcs=num_fcs,
                 ffn_drop=drop_rate,
                 dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                 if drop_path_rate > 0 else None,
                 act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as cp
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x