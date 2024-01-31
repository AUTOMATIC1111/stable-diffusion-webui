# --------------------------------------------------------
# Neighborhood Attention Transformer
# Licensed under The MIT License
# Written by Ali Hassani
# --------------------------------------------------------

# Modified by Jitesh Jain

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from annotator.oneformer.detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()


    def forward(self, x):

        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7, dilation=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, dilation=dilation, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x



class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size,  dilations=None,
                 downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads,
                     kernel_size=kernel_size,
                     dilation=None if dilations is None else dilations[i],
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class DiNAT(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 depths,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 kernel_size=7,
                 dilations=None,
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 frozen_stages=-1,
                 layer_scale=None,
                 **kwargs):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             dilations=None if dilations is None else dilations[i],
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),
                             layer_scale=layer_scale)
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(DiNAT, self).train(mode)
        self._freeze_stages()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = {}
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(xo)
                outs["res{}".format(idx + 2)] = x_out.permute(0, 3, 1, 2).contiguous()
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)


@BACKBONE_REGISTRY.register()
class D2DiNAT(DiNAT, Backbone):
    def __init__(self, cfg, input_shape):
        
        embed_dim = cfg.MODEL.DiNAT.EMBED_DIM
        mlp_ratio = cfg.MODEL.DiNAT.MLP_RATIO
        depths = cfg.MODEL.DiNAT.DEPTHS
        num_heads = cfg.MODEL.DiNAT.NUM_HEADS
        drop_path_rate = cfg.MODEL.DiNAT.DROP_PATH_RATE
        kernel_size = cfg.MODEL.DiNAT.KERNEL_SIZE
        out_indices = cfg.MODEL.DiNAT.OUT_INDICES
        dilations = cfg.MODEL.DiNAT.DILATIONS

        super().__init__(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            kernel_size=kernel_size,
            out_indices=out_indices,
            dilations=dilations,
        )

        self._out_features = cfg.MODEL.DiNAT.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"DiNAT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
