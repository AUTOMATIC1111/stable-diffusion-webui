# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


def add_deeplab_config(cfg):
    """
    Add config for DeepLab.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    # DeepLab settings
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    # Backbone new configs
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    # ResNet stem type from: `basic`, `deeplab`
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"
