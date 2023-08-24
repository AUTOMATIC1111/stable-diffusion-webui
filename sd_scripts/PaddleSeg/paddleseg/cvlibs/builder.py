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

import copy
from typing import Any, Optional

import yaml
import paddle

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import utils, logger
from paddleseg.utils.utils import CachedProperty as cached_property


class Builder(object):
    """
    The base class for building components. 

    Args:
        config (Config): A Config class.
        comp_list (list, optional): A list of component classes. Default: None
    """

    def __init__(self, config: Config, comp_list: Optional[list]=None):
        super().__init__()
        self.config = config
        self.comp_list = comp_list

    def build_component(self, cfg):
        """
        Create Python object, such as model, loss, dataset, etc.
        """
        cfg = copy.deepcopy(cfg)
        if 'type' not in cfg:
            raise RuntimeError(
                "It is not possible to create a component object from {}, as 'type' is not specified.".
                format(cfg))

        class_type = cfg.pop('type')
        com_class = self.load_component_class(class_type)

        params = {}
        for key, val in cfg.items():
            if self.is_meta_type(val):
                params[key] = self.build_component(val)
            elif isinstance(val, list):
                params[key] = [
                    self.build_component(item)
                    if self.is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val

        try:
            obj = self.build_component_impl(com_class, **params)
        except Exception as e:
            if hasattr(com_class, '__name__'):
                com_name = com_class.__name__
            else:
                com_name = ''
            raise RuntimeError(
                f"Tried to create a {com_name} object, but the operation has failed. "
                "Please double check the arguments used to create the object.\n"
                f"The error message is: \n{str(e)}")

        return obj

    def build_component_impl(self, component_class, *args, **kwargs):
        return component_class(*args, **kwargs)

    def load_component_class(self, class_type):
        for com in self.comp_list:
            if class_type in com.components_dict:
                return com[class_type]
        raise RuntimeError("The specified component ({}) was not found.".format(
            class_type))

    @classmethod
    def is_meta_type(cls, obj):
        # TODO: should we define a protocol (see https://peps.python.org/pep-0544/#defining-a-protocol)
        # to make it more pythonic?
        return isinstance(obj, dict) and 'type' in obj

    @classmethod
    def show_msg(cls, name, cfg):
        msg = 'Use the following config to build {}\n'.format(name)
        msg += str(yaml.dump({name: cfg}, Dumper=utils.NoAliasDumper))
        logger.info(msg[0:-1])


class SegBuilder(Builder):
    """
    This class is responsible for building components for semantic segmentation. 
    """

    def __init__(self, config, comp_list=None):
        if comp_list is None:
            comp_list = [
                manager.MODELS, manager.BACKBONES, manager.DATASETS,
                manager.TRANSFORMS, manager.LOSSES, manager.OPTIMIZERS
            ]
        super().__init__(config, comp_list)

    @cached_property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.config.model_cfg
        assert model_cfg != {}, \
            'No model specified in the configuration file.'

        if self.config.train_dataset_cfg['type'] != 'Dataset':
            # check and synchronize the num_classes in model config and dataset class
            assert hasattr(self.train_dataset_class, 'NUM_CLASSES'), \
                'If train_dataset class is not `Dataset`, it must have `NUM_CLASSES` attr.'
            num_classes = getattr(self.train_dataset_class, 'NUM_CLASSES')
            if 'num_classes' in model_cfg:
                assert model_cfg['num_classes'] == num_classes, \
                    'The num_classes is not consistent for model config ({}) ' \
                    'and train_dataset class ({}) '.format(model_cfg['num_classes'], num_classes)
            else:
                logger.warning(
                    'Add the `num_classes` in train_dataset class to '
                    'model config. We suggest you manually set `num_classes` in model config.'
                )
                model_cfg['num_classes'] = num_classes
            # check and synchronize the in_channels in model config and dataset class
            assert hasattr(self.train_dataset_class, 'IMG_CHANNELS'), \
                'If train_dataset class is not `Dataset`, it must have `IMG_CHANNELS` attr.'
            in_channels = getattr(self.train_dataset_class, 'IMG_CHANNELS')
            x = utils.get_in_channels(model_cfg)
            if x is not None:
                assert x == in_channels, \
                    'The in_channels in model config ({}) and the img_channels in train_dataset ' \
                    'class ({}) is not consistent'.format(x, in_channels)
            else:
                model_cfg = utils.set_in_channels(model_cfg, in_channels)
                logger.warning(
                    'Add the `in_channels` in train_dataset class to '
                    'model config. We suggest you manually set `in_channels` in model config.'
                )

        self.show_msg('model', model_cfg)
        return self.build_component(model_cfg)

    @cached_property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        opt_cfg = self.config.optimizer_cfg
        assert opt_cfg != {}, \
            'No optimizer specified in the configuration file.'
        # For compatibility
        if opt_cfg['type'] == 'adam':
            opt_cfg['type'] = 'Adam'
        if opt_cfg['type'] == 'sgd':
            opt_cfg['type'] = 'SGD'
        if opt_cfg['type'] == 'SGD' and 'momentum' in opt_cfg:
            opt_cfg['type'] = 'Momentum'
            logger.info('If the type is SGD and momentum in optimizer config, '
                        'the type is changed to Momentum.')
        self.show_msg('optimizer', opt_cfg)
        opt = self.build_component(opt_cfg)
        opt = opt(self.model, self.lr_scheduler)
        return opt

    @cached_property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        lr_cfg = self.config.lr_scheduler_cfg
        assert lr_cfg != {}, \
            'No lr_scheduler specified in the configuration file.'

        use_warmup = False
        if 'warmup_iters' in lr_cfg:
            use_warmup = True
            warmup_iters = lr_cfg.pop('warmup_iters')
            assert 'warmup_start_lr' in lr_cfg, \
                "When use warmup, please set warmup_start_lr and warmup_iters in lr_scheduler"
            warmup_start_lr = lr_cfg.pop('warmup_start_lr')
            end_lr = lr_cfg['learning_rate']

        lr_type = lr_cfg.pop('type')
        if lr_type == 'PolynomialDecay':
            iters = self.config.iters - warmup_iters if use_warmup else self.config.iters
            iters = max(iters, 1)
            lr_cfg.setdefault('decay_steps', iters)

        try:
            lr_sche = getattr(paddle.optimizer.lr, lr_type)(**lr_cfg)
        except Exception as e:
            raise RuntimeError(
                "Create {} has failed. Please check lr_scheduler in config. "
                "The error message: {}".format(lr_type, e))

        if use_warmup:
            lr_sche = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr_sche,
                warmup_steps=warmup_iters,
                start_lr=warmup_start_lr,
                end_lr=end_lr)

        return lr_sche

    @cached_property
    def loss(self) -> dict:
        loss_cfg = self.config.loss_cfg
        assert loss_cfg != {}, \
            'No loss specified in the configuration file.'
        return self._build_loss('loss', loss_cfg)

    @cached_property
    def distill_loss(self) -> dict:
        loss_cfg = self.config.distill_loss_cfg
        assert loss_cfg != {}, \
            'No distill_loss specified in the configuration file.'
        return self._build_loss('distill_loss', loss_cfg)

    def _build_loss(self, loss_name, loss_cfg: dict):
        def _check_helper(loss_cfg, ignore_index):
            if 'ignore_index' not in loss_cfg:
                loss_cfg['ignore_index'] = ignore_index
                logger.warning('Add the `ignore_index` in train_dataset ' \
                    'class to {} config. We suggest you manually set ' \
                    '`ignore_index` in {} config.'.format(loss_name, loss_name)
                )
            else:
                assert loss_cfg['ignore_index'] == ignore_index, \
                    'the ignore_index in loss and train_dataset must be the same. Currently, loss ignore_index = {}, '\
                    'train_dataset ignore_index = {}'.format(loss_cfg['ignore_index'], ignore_index)

        # check and synchronize the ignore_index in model config and dataset class
        if self.config.train_dataset_cfg['type'] != 'Dataset':
            assert hasattr(self.train_dataset_class, 'IGNORE_INDEX'), \
                'If train_dataset class is not `Dataset`, it must have `IGNORE_INDEX` attr.'
            ignore_index = getattr(self.train_dataset_class, 'IGNORE_INDEX')
            for loss_cfg_i in loss_cfg['types']:
                if loss_cfg_i['type'] == 'MixedLoss':
                    for loss_cfg_j in loss_cfg_i['losses']:
                        _check_helper(loss_cfg_j, ignore_index)
                else:
                    _check_helper(loss_cfg_i, ignore_index)

        self.show_msg(loss_name, loss_cfg)
        loss_dict = {'coef': loss_cfg['coef'], "types": []}
        for item in loss_cfg['types']:
            loss_dict['types'].append(self.build_component(item))
        return loss_dict

    @cached_property
    def train_dataset(self) -> paddle.io.Dataset:
        dataset_cfg = self.config.train_dataset_cfg
        assert dataset_cfg != {}, \
            'No train_dataset specified in the configuration file.'
        self.show_msg('train_dataset', dataset_cfg)
        dataset = self.build_component(dataset_cfg)
        assert len(dataset) != 0, \
            'The number of samples in train_dataset is 0. Please check whether the dataset is valid.'
        return dataset

    @cached_property
    def val_dataset(self) -> paddle.io.Dataset:
        dataset_cfg = self.config.val_dataset_cfg
        assert dataset_cfg != {}, \
            'No val_dataset specified in the configuration file.'
        self.show_msg('val_dataset', dataset_cfg)
        dataset = self.build_component(dataset_cfg)
        if len(dataset) == 0:
            logger.warning(
                'The number of samples in val_dataset is 0. Please ensure this is the desired behavior.'
            )
        return dataset

    @cached_property
    def train_dataset_class(self) -> Any:
        dataset_cfg = self.config.train_dataset_cfg
        assert dataset_cfg != {}, \
            'No train_dataset specified in the configuration file.'
        dataset_type = dataset_cfg.get('type')
        return self.load_component_class(dataset_type)

    @cached_property
    def val_dataset_class(self) -> Any:
        dataset_cfg = self.config.val_dataset_cfg
        assert dataset_cfg != {}, \
            'No val_dataset specified in the configuration file.'
        dataset_type = dataset_cfg.get('type')
        return self.load_component_class(dataset_type)

    @cached_property
    def val_transforms(self) -> list:
        dataset_cfg = self.config.val_dataset_cfg
        assert dataset_cfg != {}, \
            'No val_dataset specified in the configuration file.'
        transforms = []
        for item in dataset_cfg.get('transforms', []):
            transforms.append(self.build_component(item))
        return transforms
