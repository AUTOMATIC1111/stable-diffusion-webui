# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from torch.nn import GroupNorm, LayerNorm

from annotator.mmpkg.mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from annotator.mmpkg.mmcv.utils.ext_loader import check_ops_exist
from .builder import OPTIMIZER_BUILDERS, OPTIMIZERS


@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructor:
    """Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers and offset layers of DCN).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers, depthwise conv layers, offset layers of DCN).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
      rate for parameters of offset layer in the deformable convs
      of a model.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Default: False.

    Note:
        1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
            override the effect of ``bias_lr_mult`` in the bias of offset
            layer. So be careful when using both ``bias_lr_mult`` and
            ``dcn_offset_lr_mult``. If you wish to apply both of them to the
            offset layer in deformable convs, set ``dcn_offset_lr_mult``
            to the original ``dcn_offset_lr_mult`` * ``bias_lr_mult``.
        2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
            apply it to all the DCN layers in the model. So be careful when
            the model contains multiple DCN layers in places other than
            backbone.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
        >>> paramwise_cfg = dict(custom_keys={
                '.backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg.get('lr', None)
        self.base_wd = optimizer_cfg.get('weight_decay', None)
        self._validate_cfg()

    def _validate_cfg(self):
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, '
                            f'but got {type(self.paramwise_cfg)}')

        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def _is_in(self, param_group, param_group_list):
        assert is_list_of(param_group_list, dict)
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)

        if check_ops_exist():
            from annotator.mmpkg.mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
