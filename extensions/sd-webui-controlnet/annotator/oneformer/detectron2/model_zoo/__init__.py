# Copyright (c) Facebook, Inc. and its affiliates.
"""
Model Zoo API for Detectron2: a collection of functions to create common model architectures
listed in `MODEL_ZOO.md <https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md>`_,
and optionally load their pre-trained weights.
"""

from .model_zoo import get, get_config_file, get_checkpoint_url, get_config

__all__ = ["get_checkpoint_url", "get", "get_config_file", "get_config"]
