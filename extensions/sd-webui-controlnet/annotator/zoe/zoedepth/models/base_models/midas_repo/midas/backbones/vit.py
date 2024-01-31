import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F

from .utils import (activations, forward_adapted_unflatten, get_activation, get_readout_oper,
                    make_backbone_default, Transpose)


def forward_vit(pretrained, x):
    return forward_adapted_unflatten(pretrained, x, "forward_flex")


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index:],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        if self.no_embed_class:
            x = x + pos_embed
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    if not self.no_embed_class:
        x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    start_index_readout=1,
):
    pretrained = make_backbone_default(model, features, size, hooks, vit_features, use_readout, start_index,
                                       start_index_readout)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitl16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def _make_pretrained_vitb16_384(pretrained, use_readout="ignore", hooks=None):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model, features=[96, 192, 384, 768], hooks=hooks, use_readout=use_readout
    )


def _make_vit_b_rn50_backbone(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    patch_size=[16, 16],
    number_stages=2,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model

    used_number_stages = 0 if use_vit_only else number_stages
    for s in range(used_number_stages):
        pretrained.model.patch_embed.backbone.stages[s].register_forward_hook(
            get_activation(str(s + 1))
        )
    for s in range(used_number_stages, 4):
        pretrained.model.blocks[hooks[s]].register_forward_hook(get_activation(str(s + 1)))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    for s in range(used_number_stages):
        value = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity())
        exec(f"pretrained.act_postprocess{s + 1}=value")
    for s in range(used_number_stages, 4):
        if s < number_stages:
            final_layer = nn.ConvTranspose2d(
                in_channels=features[s],
                out_channels=features[s],
                kernel_size=4 // (2 ** s),
                stride=4 // (2 ** s),
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif s > number_stages:
            final_layer = nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            final_layer = None

        layers = [
            readout_oper[s],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[s],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        ]
        if final_layer is not None:
            layers.append(final_layer)

        value = nn.Sequential(*layers)
        exec(f"pretrained.act_postprocess{s + 1}=value")

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = patch_size

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384(
    pretrained, use_readout="ignore", hooks=None, use_vit_only=False
):
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
    )
