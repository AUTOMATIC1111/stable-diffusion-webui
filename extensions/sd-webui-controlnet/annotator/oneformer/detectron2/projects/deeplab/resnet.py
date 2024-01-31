# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from annotator.oneformer.detectron2.layers import CNNBlockBase, Conv2d, get_norm
from annotator.oneformer.detectron2.modeling import BACKBONE_REGISTRY
from annotator.oneformer.detectron2.modeling.backbone.resnet import (
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock,
    ResNet,
)


class DeepLabStem(CNNBlockBase):
    """
    The DeepLab ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=128, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv2 = Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv3 = Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


@BACKBONE_REGISTRY.register()
def build_resnet_deeplab_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    if cfg.MODEL.RESNETS.STEM_TYPE == "basic":
        stem = BasicStem(
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    elif cfg.MODEL.RESNETS.STEM_TYPE == "deeplab":
        stem = DeepLabStem(
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    else:
        raise ValueError("Unknown stem type: {}".format(cfg.MODEL.RESNETS.STEM_TYPE))

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res4_dilation       = cfg.MODEL.RESNETS.RES4_DILATION
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    res5_multi_grid     = cfg.MODEL.RESNETS.RES5_MULTI_GRID
    # fmt: on
    assert res4_dilation in {1, 2}, "res4_dilation cannot be {}.".format(res4_dilation)
    assert res5_dilation in {1, 2, 4}, "res5_dilation cannot be {}.".format(res5_dilation)
    if res4_dilation == 2:
        # Always dilate res5 if res4 is dilated.
        assert res5_dilation == 4

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        if stage_idx == 4:
            dilation = res4_dilation
        elif stage_idx == 5:
            dilation = res5_dilation
        else:
            dilation = 1
        first_stride = 1 if idx == 0 or dilation > 1 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        if stage_idx == 5:
            stage_kargs.pop("dilation")
            stage_kargs["dilation_per_block"] = [dilation * mg for mg in res5_multi_grid]
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)
