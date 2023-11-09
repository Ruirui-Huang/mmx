import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init, trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.models.utils import resize

from mmx.registry import MODELS


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    # rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    # rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # TODO:模型转换tips，pytorch->onnx->other时，替换other暂时不支持的Einsum操作，将torch.einsum替换为torch.matmul
    rel_h = torch.matmul(r_q, Rh.transpose(1, 2))
    rel_w = torch.matmul(r_q.transpose(1, 2), Rw.transpose(1, 2)).transpose(1, 2)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, in_channels=3, embed_dims=768, kernel_size=16, stride=16, padding=0):
        """
        Args:
            in_channels (int): Number of input image channels.
            embed_dims (int):  embed_dims (int): Patch embedding dimension.
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
        """
        super().__init__()

        self.proj = nn.Conv2d(in_channels,
                              embed_dims,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        x = self.proj(x)
        # out_size = (x.shape[2], x.shape[3])
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(BaseModule):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 input_size=None):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(BaseModule):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at
            the last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_channels: int,
                 stride: int = 1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 act_cfg_out=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(in_channels,
                                bottleneck_channels,
                                1,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv2 = ConvModule(bottleneck_channels,
                                bottleneck_channels,
                                3,
                                stride,
                                1,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv3 = ConvModule(bottleneck_channels,
                                out_channels,
                                1,
                                norm_cfg=norm_cfg,
                                act_cfg=None)
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out


class VitEncoderLayer(BaseModule):
    """Transformer encoder layer with support of window attention and residual propagation blocks"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=0,
                 use_residual_block=False,
                 input_size=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 ffn_cfg=dict(),
                 with_cp=False):
        """
        Args:
            embed_dims (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)

        self.add_module(self.norm1_name, norm1)
        self.attn = Attention(embed_dims,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              use_rel_pos=use_rel_pos,
                              rel_pos_zero_init=rel_pos_zero_init,
                              input_size=input_size if window_size == 0 else
                              (window_size, window_size))

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        dropout_layer = dict(type='DropPath',
                             drop_prob=drop_path_rate) if drop_path_rate > 0 else None
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        ffn_cfg.update(
            dict(embed_dims=embed_dims,
                 feedforward_channels=feedforward_channels,
                 num_fcs=num_fcs,
                 ffn_drop=drop_rate,
                 dropout_layer=dropout_layer,
                 act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)

        self.with_cp = with_cp

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(in_channels=embed_dims,
                                               out_channels=embed_dims,
                                               bottleneck_channels=embed_dims // 2,
                                               norm_cfg=norm_cfg,
                                               act_layer=act_cfg)

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
            shortcut = x
            x = self.norm1(x)
            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x)
            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))

            x = shortcut + self.dropout_layer(x)
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
            x = self.ffn(self.norm2(x), identity=x)

            if self.use_residual_block:
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


@MODELS.register_module()
class VisionTransformer(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed VitEncoderLayer.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=0,
                 window_block_indexes=(),
                 residual_block_indexes=(),
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrain_use_cls_token=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(in_channels=in_channels,
                                      embed_dims=embed_dims,
                                      kernel_size=patch_size,
                                      stride=patch_size)

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches

        # if use_abs_pos:
        #     # Initialize absolute positional embedding with pretrain image size.
        #     num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        #     num_positions = (num_patches + 1) if with_cls_token else num_patches
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dims))
        # else:
        #     self.pos_embed = None

        self.with_cls_token = pretrain_use_cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
               ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                VitEncoderLayer(embed_dims=embed_dims,
                                num_heads=num_heads,
                                feedforward_channels=mlp_ratio * embed_dims,
                                drop_rate=drop_rate,
                                drop_path_rate=dpr[i],
                                num_fcs=num_fcs,
                                qkv_bias=qkv_bias,
                                use_rel_pos=use_rel_pos,
                                rel_pos_zero_init=rel_pos_zero_init,
                                window_size=window_size if i in window_block_indexes else 0,
                                use_residual_block=i in residual_block_indexes,
                                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
                                act_cfg=act_cfg,
                                norm_cfg=norm_cfg,
                                with_cp=with_cp))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict) and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = CheckpointLoader.load_checkpoint(self.init_cfg['checkpoint'],
                                                          logger=None,
                                                          map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    pos_embed_weight = state_dict['pos_embed']
                    if self.with_cls_token:
                        cls_token_weight = state_dict['pos_embed'][:, 0]
                        pos_embed_weight = state_dict['pos_embed'][:, 1:]
                    pos_embed_weight = self.resize_pos_embed(
                        pos_embed_weight, (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)
                    if self.with_cls_token:
                        state_dict['pos_embed'] = torch.cat(
                            (cls_token_weight.unsqueeze(1), pos_embed_weight), dim=1)
                    else:
                        state_dict['pos_embed'] = pos_embed_weight

            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            # trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 4, 'the shapes of patched_img must be [B, H, W, C]'
        assert pos_embed.ndim == 3, 'the shapes of pos_embed must be [B, L, C]'
        x_len = patched_img.shape[1] * patched_img.shape[2]
        pos_len = pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == x_len - 1:
                pos_embed_4dim = pos_embed.reshape(1, hw_shape[0], hw_shape[1], -1)
            elif pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] //
                                                                     self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError('Unexpected shape of pos_embed, got {}.'.format(pos_embed.shape))
            # 不考虑处理cls_token
            pos_embed = pos_embed[:, 1:]
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape, (pos_h, pos_w),
                                              self.interpolate_mode)
            pos_embed_4dim = pos_embed.reshape(1, hw_shape[0], hw_shape[1], -1)
        else:
            pos_embed_4dim = pos_embed.reshape(1, hw_shape[0], hw_shape[1], -1)
        return self.drop_after_pos(patched_img + pos_embed_4dim)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        pos_embed = pos_embed.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed = resize(pos_embed, size=input_shpae, align_corners=False, mode=mode)
        pos_embed = torch.flatten(pos_embed, 2).transpose(1, 2)
        return pos_embed

    def forward(self, inputs):
        x = self.patch_embed(inputs)
        x = self._pos_embeding(x, (x.shape[1], x.shape[2]), self.pos_embed)
        # x = x + self.pos_embed[:, 1:].reshape(1, x.shape[1], x.shape[2], -1)

        # if self.pos_embed is not None:
        #     x = x + get_abs_pos(self.pos_embed, self.with_cls_token, (x.shape[1], x.shape[2]))

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                out = x.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()