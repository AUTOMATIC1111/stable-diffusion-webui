import logging
import numpy as np
import torch
import torch.nn as nn

from .backbone import Backbone
from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)

logger = logging.getLogger(__name__)


__all__ = ["MViT"]


def attention_pool(x, pool, norm=None):
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H1, W1) -> (B, H1, W1, C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """Multiscale Multi-head Attention block."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # qkv pooling
        pool_padding = [k // 2 for k in pool_kernel]
        dim_conv = dim_out // num_heads
        self.pool_q = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_q,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = norm_layer(dim_conv)
        self.pool_k = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_k = norm_layer(dim_conv)
        self.pool_v = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_v = norm_layer(dim_conv)

        self.window_size = window_size
        if window_size:
            self.q_win_size = window_size // stride_q
            self.kv_win_size = window_size // stride_kv
        self.residual_pooling = residual_pooling

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            assert input_size[0] == input_size[1]
            size = input_size[0]
            rel_dim = 2 * max(size // stride_q, size // stride_kv) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H, W, C)
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        # q, k, v with shape (B * nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q)
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        ori_q = q
        if self.window_size:
            q, q_hw_pad = window_partition(q, self.q_win_size)
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw)

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(x, self.q_win_size, q_hw_pad, ori_q.shape[1:3])

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """Multiscale Transformer blocks"""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if stride_q > 1:
            kernel_skip = stride_q + 1
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_q, padding_skip, ceil_mode=False)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_block = self.attn(x_norm)

        if hasattr(self, "proj"):
            x = self.proj(x_norm)
        if hasattr(self, "pool_skip"):
            x = attention_pool(x, self.pool_skip)

        x = x + self.drop_path(x_block)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MViT(Backbone):
    """
    This module implements Multiscale Vision Transformer (MViT) backbone in :paper:'mvitv2'.
    """

    def __init__(
        self,
        img_size=224,
        patch_kernel=(7, 7),
        patch_stride=(4, 4),
        patch_padding=(3, 3),
        in_chans=3,
        embed_dim=96,
        depth=16,
        num_heads=1,
        last_block_indexes=(0, 2, 11, 15),
        qkv_pool_kernel=(3, 3),
        adaptive_kv_stride=4,
        adaptive_window_size=56,
        residual_pooling=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=False,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_features=("scale2", "scale3", "scale4", "scale5"),
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_kernel (tuple): kernel size for patch embedding.
            patch_stride (tuple): stride size for patch embedding.
            patch_padding (tuple): padding size for patch embedding.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of MViT.
            num_heads (int): Number of base attention heads in each MViT block.
            last_block_indexes (tuple): Block indexes for last blocks in each stage.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            adaptive_kv_stride (int): adaptive stride size for kv pooling.
            adaptive_window_size (int): adaptive window size for window attention blocks.
            residual_pooling (bool): If true, enable residual pooling.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_features (tuple): name of the feature maps from each stage.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absoluate positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_stride[0]) * (
                pretrain_img_size // patch_stride[1]
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dim_out = embed_dim
        stride_kv = adaptive_kv_stride
        window_size = adaptive_window_size
        input_size = (img_size // patch_stride[0], img_size // patch_stride[1])
        stage = 2
        stride = patch_stride[0]
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Multiply stride_kv by 2 if it's the last block of stage2 and stage3.
            if i == last_block_indexes[1] or i == last_block_indexes[2]:
                stride_kv_ = stride_kv * 2
            else:
                stride_kv_ = stride_kv
            # hybrid window attention: global attention in last three stages.
            window_size_ = 0 if i in last_block_indexes[1:] else window_size
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qkv_pool_kernel=qkv_pool_kernel,
                stride_q=2 if i - 1 in last_block_indexes else 1,
                stride_kv=stride_kv_,
                residual_pooling=residual_pooling,
                window_size=window_size_,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
            )
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

            embed_dim = dim_out
            if i in last_block_indexes:
                name = f"scale{stage}"
                if name in out_features:
                    self._out_feature_channels[name] = dim_out
                    self._out_feature_strides[name] = stride
                    self.add_module(f"{name}_norm", norm_layer(dim_out))

                dim_out *= 2
                num_heads *= 2
                stride_kv = max(stride_kv // 2, 1)
                stride *= 2
                stage += 1
            if i - 1 in last_block_indexes:
                window_size = window_size // 2
                input_size = [s // 2 for s in input_size]

        self._out_features = out_features
        self._last_block_indexes = last_block_indexes

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, x.shape[1:3])

        outputs = {}
        stage = 2
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self._last_block_indexes:
                name = f"scale{stage}"
                if name in self._out_features:
                    x_out = getattr(self, f"{name}_norm")(x)
                    outputs[name] = x_out.permute(0, 3, 1, 2)
                stage += 1

        return outputs
