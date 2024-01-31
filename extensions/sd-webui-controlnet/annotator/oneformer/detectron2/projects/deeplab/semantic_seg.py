# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.layers import ASPP, Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from annotator.oneformer.detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .loss import DeepLabCE


@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabV3PlusHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        train_size: Optional[Tuple],
        loss_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        ignore_value: int = -1,
        num_classes: Optional[int] = None,
        use_depthwise_separable_conv: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shape of the input features. They will be ordered by stride
                and the last one (with largest stride) is used as the input to the
                decoder (i.e.  the ASPP module); the rest are low-level feature for
                the intermediate levels of decoder.
            project_channels (list[int]): a list of low-level feature channels.
                The length should be len(in_features) - 1.
            aspp_dilations (list(int)): a list of 3 dilations in ASPP.
            aspp_dropout (float): apply dropout on the output of ASPP.
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            common_stride (int): output stride of decoder.
            norm (str or callable): normalization for all conv layers.
            train_size (tuple): (height, width) of training images.
            loss_weight (float): loss weight.
            loss_type (str): type of loss function, 2 opptions:
                (1) "cross_entropy" is the standard cross entropy loss.
                (2) "hard_pixel_mining" is the loss in DeepLab that samples
                    top k% hardest pixels.
            ignore_value (int): category to be ignored during training.
            num_classes (int): number of classes, if set to None, the decoder
                will not construct a predictor.
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                in ASPP and decoder.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features      = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels           = [x[1].channels for x in input_shape]
        in_strides            = [x[1].stride for x in input_shape]
        aspp_channels         = decoder_channels[-1]
        self.ignore_value     = ignore_value
        self.common_stride    = common_stride  # output stride
        self.loss_weight      = loss_weight
        self.loss_type        = loss_type
        self.decoder_only     = num_classes is None
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        # fmt: on

        assert (
            len(project_channels) == len(self.in_features) - 1
        ), "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(
            self.in_features
        ), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        self.decoder = nn.ModuleDict()

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            decoder_stage = nn.ModuleDict()

            if idx == len(self.in_features) - 1:
                # ASPP module
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    if train_h % encoder_stride or train_w % encoder_stride:
                        raise ValueError("Crop size need to be divisible by encoder stride.")
                    pool_h = train_h // encoder_stride
                    pool_w = train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None
                project_conv = ASPP(
                    in_channel,
                    aspp_channels,
                    aspp_dilations,
                    norm=norm,
                    activation=F.relu,
                    pool_kernel_size=pool_kernel_size,
                    dropout=aspp_dropout,
                    use_depthwise_separable_conv=use_depthwise_separable_conv,
                )
                fuse_conv = None
            else:
                project_conv = Conv2d(
                    in_channel,
                    project_channels[idx],
                    kernel_size=1,
                    bias=use_bias,
                    norm=get_norm(norm, project_channels[idx]),
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(project_conv)
                if use_depthwise_separable_conv:
                    # We use a single 5x5 DepthwiseSeparableConv2d to replace
                    # 2 3x3 Conv2d since they have the same receptive field,
                    # proposed in :paper:`Panoptic-DeepLab`.
                    fuse_conv = DepthwiseSeparableConv2d(
                        project_channels[idx] + decoder_channels[idx + 1],
                        decoder_channels[idx],
                        kernel_size=5,
                        padding=2,
                        norm1=norm,
                        activation1=F.relu,
                        norm2=norm,
                        activation2=F.relu,
                    )
                else:
                    fuse_conv = nn.Sequential(
                        Conv2d(
                            project_channels[idx] + decoder_channels[idx + 1],
                            decoder_channels[idx],
                            kernel_size=3,
                            padding=1,
                            bias=use_bias,
                            norm=get_norm(norm, decoder_channels[idx]),
                            activation=F.relu,
                        ),
                        Conv2d(
                            decoder_channels[idx],
                            decoder_channels[idx],
                            kernel_size=3,
                            padding=1,
                            bias=use_bias,
                            norm=get_norm(norm, decoder_channels[idx]),
                            activation=F.relu,
                        ),
                    )
                    weight_init.c2_xavier_fill(fuse_conv[0])
                    weight_init.c2_xavier_fill(fuse_conv[1])

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[self.in_features[idx]] = decoder_stage

        if not self.decoder_only:
            self.predictor = Conv2d(
                decoder_channels[0], num_classes, kernel_size=1, stride=1, padding=0
            )
            nn.init.normal_(self.predictor.weight, 0, 0.001)
            nn.init.constant_(self.predictor.bias, 0)

            if self.loss_type == "cross_entropy":
                self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
            elif self.loss_type == "hard_pixel_mining":
                self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
            else:
                raise ValueError("Unexpected loss type: %s" % self.loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            train_size=train_size,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.decoder_only:
            # Output from self.layers() only contains decoder feature.
            return y
        if self.training:
            return None, self.losses(y, targets)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for f in self.in_features[::-1]:
            x = features[f]
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)
        if not self.decoder_only:
            y = self.predictor(y)
        return y

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabV3Head(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        in_channels           = [input_shape[f].channels for f in self.in_features]
        aspp_channels         = cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS
        aspp_dilations        = cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE  # output stride
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.loss_type        = cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE
        train_crop_size       = cfg.INPUT.CROP.SIZE
        aspp_dropout          = cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT
        use_depthwise_separable_conv = cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
        # fmt: on

        assert len(self.in_features) == 1
        assert len(in_channels) == 1

        # ASPP module
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_crop_h, train_crop_w = train_crop_size
            if train_crop_h % self.common_stride or train_crop_w % self.common_stride:
                raise ValueError("Crop size need to be divisible by output stride.")
            pool_h = train_crop_h // self.common_stride
            pool_w = train_crop_w // self.common_stride
            pool_kernel_size = (pool_h, pool_w)
        else:
            pool_kernel_size = None
        self.aspp = ASPP(
            in_channels[0],
            aspp_channels,
            aspp_dilations,
            norm=norm,
            activation=F.relu,
            pool_kernel_size=pool_kernel_size,
            dropout=aspp_dropout,
            use_depthwise_separable_conv=use_depthwise_separable_conv,
        )

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if self.loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif self.loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = features[self.in_features[0]]
        x = self.aspp(x)
        x = self.predictor(x)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
