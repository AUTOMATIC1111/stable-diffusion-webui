# Copyright (c) OpenMMLab. All rights reserved.
from ..runner import Sequential
from ..utils import Registry, build_from_cfg


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


MODELS = Registry('model', build_func=build_model_from_cfg)
