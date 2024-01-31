# Copyright (c) Facebook, Inc. and its affiliates.
from .build_solver import build_lr_scheduler
from .config import add_deeplab_config
from .resnet import build_resnet_deeplab_backbone
from .semantic_seg import DeepLabV3Head, DeepLabV3PlusHead
