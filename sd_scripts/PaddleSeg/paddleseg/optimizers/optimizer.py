# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from paddleseg.cvlibs import manager
from paddleseg.utils import logger
import paddleseg.optimizers.custom_optimizers as custom_opt


class BaseOptimizer(object):
    """
    Base optimizer in PaddleSeg.

    Args:
        weight_decay(float, optional): A float value as coeff of L2 regularization.
        grad_clip_cfg(dict, optional): A dict to specify grad_clip. It must have the following format: 
            {'name': 'ClipGradByGlobalNorm', 'clip_norm': float_val},
            {'name': 'ClipGradByNorm', 'clip_norm': float_val},
            {'name': 'ClipGradByValue', 'max': float_val, 'min': float_val(optional)}.
        custom_cfg(list, optional): custom_cfg specify different options for
            different parameter groups such as the learning rate and weight decay.
            For example, [{'name': 'backbone', 'lr_mult': 0.1}, {'name': 'norm', 'weight_decay_mult': 0}]
    
    An example in config:
    `
    optimizer:
      type: SGD
      weight_decay: 4.0e-5
      custom_cfg:
        - name: backbone
          lr_mult: 0.1
        - name: norm
          weight_decay_mult: 0.0
      grad_clip_cfg:
        name: ClipGradByValue
        max: 1.0
    `
    """

    def __init__(self, weight_decay=None, grad_clip_cfg=None, custom_cfg=None):
        if weight_decay is not None:
            assert isinstance(weight_decay, float), \
                "`weight_decay` must be a float."
        if grad_clip_cfg is not None:
            assert isinstance(grad_clip_cfg, dict), \
                "`grad_clip_cfg` must be a dict."
            assert 'name' in grad_clip_cfg, 'No name specified in grad_clip_cfg'
            grad_clip_names = [
                'ClipGradByValue', 'ClipGradByNorm', 'ClipGradByGlobalNorm'
            ]
            assert grad_clip_cfg['name'] in grad_clip_names, \
                'grad_clip name should be {}'.format(grad_clip_names)
        if custom_cfg is not None:
            assert isinstance(custom_cfg, list), "`custom_cfg` must be a list."
            for item in custom_cfg:
                assert isinstance(
                    item, dict), "The item of `custom_cfg` must be a dict"

        self.weight_decay = weight_decay
        self.custom_cfg = custom_cfg
        self.args = {'weight_decay': weight_decay}

        if grad_clip_cfg is not None:
            grad_clip_cfg = grad_clip_cfg.copy()
            grad_clip_name = grad_clip_cfg.pop('name')
            try:
                grad_clip = getattr(paddle.nn, grad_clip_name)(**grad_clip_cfg)
            except Exception as e:
                raise RuntimeError(
                    "Create grad_clip has failed. Please check grad_clip_cfg in config. "
                    f"The error message: \n{str(e)}")
            self.args.update({'grad_clip': grad_clip})

    def __call__(self, model, lr):
        # Create optimizer
        pass

    def _collect_params(self, model):
        # Collect different parameter groups
        if self.custom_cfg is None or len(self.custom_cfg) == 0:
            return model.parameters()

        groups_num = len(self.custom_cfg) + 1
        params_list = [[] for _ in range(groups_num)]
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            for idx, item in enumerate(self.custom_cfg):
                if item['name'] in name:
                    params_list[idx].append(param)
                    break
            else:
                params_list[-1].append(param)

        res = []
        for idx, item in enumerate(self.custom_cfg):
            lr_mult = item.get("lr_mult", 1.0)
            weight_decay_mult = item.get("weight_decay_mult", None)
            param_dict = {'params': params_list[idx], 'learning_rate': lr_mult}
            if self.weight_decay is not None and weight_decay_mult is not None:
                param_dict[
                    'weight_decay'] = self.weight_decay * weight_decay_mult
            res.append(param_dict)
        res.append({'params': params_list[-1]})

        msg = 'Parameter groups for optimizer: \n'
        for idx, item in enumerate(self.custom_cfg):
            params_name = [p.name for p in params_list[idx]]
            item = item.copy()
            item['params_name'] = params_name
            msg += 'Group {}: \n{} \n'.format(idx, item)
        msg += 'Last group:\n params_name: {}'.format(
            [p.name for p in params_list[-1]])
        logger.info(msg)

        return res


@manager.OPTIMIZERS.add_component
class SGD(BaseOptimizer):
    """
    SGD optimizer. 

    An example in config:
    `
    optimizer:
      type: SGD
      weight_decay: 4.0e-5
      custom_cfg:
        - name: backbone
          lr_mult: 0.1
        - name: norm
          weight_decay_mult: 0.0
    `
    """

    def __init__(self, weight_decay=None, grad_clip_cfg=None, custom_cfg=None):
        super().__init__(weight_decay, grad_clip_cfg, custom_cfg)

    def __call__(self, model, lr):
        params = self._collect_params(model)
        return paddle.optimizer.SGD(learning_rate=lr,
                                    parameters=params,
                                    **self.args)


@manager.OPTIMIZERS.add_component
class Momentum(BaseOptimizer):
    """
    Momentum optimizer. 
    """

    def __init__(self,
                 momentum=0.9,
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip_cfg=None,
                 custom_cfg=None):
        super().__init__(weight_decay, grad_clip_cfg, custom_cfg)
        self.args.update({'momentum': momentum, 'use_nesterov': use_nesterov})

    def __call__(self, model, lr):
        params = self._collect_params(model)
        return paddle.optimizer.Momentum(
            learning_rate=lr, parameters=params, **self.args)


@manager.OPTIMIZERS.add_component
class Adam(BaseOptimizer):
    """
    Adam optimizer. 
    """

    def __init__(self,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 lazy_mode=False,
                 weight_decay=None,
                 grad_clip_cfg=None,
                 custom_cfg=None):
        super().__init__(weight_decay, grad_clip_cfg, custom_cfg)
        self.args.update({
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode
        })

    def __call__(self, model, lr):
        params = self._collect_params(model)
        opt = paddle.optimizer.Adam(
            learning_rate=lr, parameters=params, **self.args)
        return opt


@manager.OPTIMIZERS.add_component
class AdamW(BaseOptimizer):
    """
    AdamW optimizer. 
    """

    def __init__(self,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 weight_decay=0.01,
                 lazy_mode=False,
                 grad_clip_cfg=None,
                 custom_cfg=None):
        super().__init__(weight_decay, grad_clip_cfg, custom_cfg)
        self.args.update({
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode
        })

    def __call__(self, model, lr):
        params = self._collect_params(model)
        opt = paddle.optimizer.AdamW(
            learning_rate=lr, parameters=params, **self.args)
        return opt


@manager.OPTIMIZERS.add_component
class AdamWDL(BaseOptimizer):
    """
    AdamW optimizer. 
    """

    def __init__(self,
                 beta1=0.9,
                 beta2=0.999,
                 weight_decay=0.01,
                 layerwise_decay=0.65,
                 lazy_mode=False,
                 grad_clip_cfg=None,
                 custom_cfg=None):
        super().__init__(weight_decay, grad_clip_cfg, custom_cfg)
        self.args.update({
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'layerwise_decay': layerwise_decay,
            'lazy_mode': lazy_mode
        })

    def __call__(self, model, lr):
        params = self._collect_params(model)
        opt = custom_opt.AdamWDL(
            learning_rate=lr, parameters=params, **self.args)
        return opt


@manager.OPTIMIZERS.add_component
class AdamWDL_CAE(AdamWDL):
    """
    AdamW optimizer. 
    """

    def __init__(self,
                 beta1=0.9,
                 beta2=0.999,
                 weight_decay=0.01,
                 layerwise_decay=0.65,
                 lazy_mode=False,
                 grad_clip_cfg=None,
                 custom_cfg=None):
        super().__init__(weight_decay, grad_clip_cfg, custom_cfg)
        self.args.update({
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'layerwise_decay': layerwise_decay,
            'lazy_mode': lazy_mode
        })

    def __call__(self, model, lr):
        params = self._collect_params(model)
        skip_list = model.backbone.no_weight_decay()

        decay_dict = {
            param.name: not (len(param.shape) == 1 or name.endswith(".bias") or
                             name in skip_list)
            for name, param in model.named_parameters()
        }
        self.args['n_layers'] = model.backbone.get_num_layers()
        self.args['apply_decay_param_fun'] = lambda n: decay_dict[n]
        name_dict = dict()
        for n, p in model.named_parameters():
            name_dict[p.name] = n
        self.args['name_dict'] = name_dict

        opt = custom_opt.AdamWDL(
            learning_rate=lr, parameters=params, **self.args)
        return opt
