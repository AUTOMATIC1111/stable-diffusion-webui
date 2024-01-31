import copy
import logging
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DistributedSampler

# from annotator.lama.saicinpainting.evaluation import make_evaluator
# from annotator.lama.saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
# from annotator.lama.saicinpainting.training.losses.adversarial import make_discrim_loss
# from annotator.lama.saicinpainting.training.losses.perceptual import PerceptualLoss, ResNetPL
from annotator.lama.saicinpainting.training.modules import make_generator  #, make_discriminator
# from annotator.lama.saicinpainting.training.visualizers import make_visualizer
from annotator.lama.saicinpainting.utils import add_prefix_to_keys, average_dicts, set_requires_grad, flatten_dict, \
    get_has_ddp_rank

LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def update_running_average(result: nn.Module, new_iterate_model: nn.Module, decay=0.999):
    with torch.no_grad():
        res_params = dict(result.named_parameters())
        new_params = dict(new_iterate_model.named_parameters())

        for k in res_params.keys():
            res_params[k].data.mul_(decay).add_(new_params[k].data, alpha=1 - decay)


def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
    for _ in range(scales):
        cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
        cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


class BaseInpaintingTrainingModule(ptl.LightningModule):
    def __init__(self, config, use_ddp, *args,  predict_only=False, visualize_each_iters=100,
                 average_generator=False, generator_avg_beta=0.999, average_generator_start_step=30000,
                 average_generator_period=10, store_discr_outputs_for_vis=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        LOGGER.info('BaseInpaintingTrainingModule init called')

        self.config = config

        self.generator = make_generator(config, **self.config.generator)
        self.use_ddp = use_ddp

        if not get_has_ddp_rank():
            LOGGER.info(f'Generator\n{self.generator}')

        # if not predict_only:
        #     self.save_hyperparameters(self.config)
        #     self.discriminator = make_discriminator(**self.config.discriminator)
        #     self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)
        #     self.visualizer = make_visualizer(**self.config.visualizer)
        #     self.val_evaluator = make_evaluator(**self.config.evaluator)
        #     self.test_evaluator = make_evaluator(**self.config.evaluator)
        #
        #     if not get_has_ddp_rank():
        #         LOGGER.info(f'Discriminator\n{self.discriminator}')
        #
        #     extra_val = self.config.data.get('extra_val', ())
        #     if extra_val:
        #         self.extra_val_titles = list(extra_val)
        #         self.extra_evaluators = nn.ModuleDict({k: make_evaluator(**self.config.evaluator)
        #                                                for k in extra_val})
        #     else:
        #         self.extra_evaluators = {}
        #
        #     self.average_generator = average_generator
        #     self.generator_avg_beta = generator_avg_beta
        #     self.average_generator_start_step = average_generator_start_step
        #     self.average_generator_period = average_generator_period
        #     self.generator_average = None
        #     self.last_generator_averaging_step = -1
        #     self.store_discr_outputs_for_vis = store_discr_outputs_for_vis
        #
        #     if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
        #         self.loss_l1 = nn.L1Loss(reduction='none')
        #
        #     if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
        #         self.loss_mse = nn.MSELoss(reduction='none')
        #
        #     if self.config.losses.perceptual.weight > 0:
        #         self.loss_pl = PerceptualLoss()
        #
        #     # if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
        #     #     self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)
        #     # else:
        #     #     self.loss_resnet_pl = None
        #
        #     self.loss_resnet_pl = None

        self.visualize_each_iters = visualize_each_iters
        LOGGER.info('BaseInpaintingTrainingModule init done')

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
            dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
        ]

    def train_dataloader(self):
        kwargs = dict(self.config.data.train)
        if self.use_ddp:
            kwargs['ddp_kwargs'] = dict(num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
                                        rank=self.trainer.global_rank,
                                        shuffle=True)
        dataloader = make_default_train_dataloader(**self.config.data.train)
        return dataloader

    def val_dataloader(self):
        res = [make_default_val_dataloader(**self.config.data.val)]

        if self.config.data.visual_test is not None:
            res = res + [make_default_val_dataloader(**self.config.data.visual_test)]
        else:
            res = res + res

        extra_val = self.config.data.get('extra_val', ())
        if extra_val:
            res += [make_default_val_dataloader(**extra_val[k]) for k in self.extra_val_titles]

        return res

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self._is_training_step = True
        return self._do_step(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        extra_val_key = None
        if dataloader_idx == 0:
            mode = 'val'
        elif dataloader_idx == 1:
            mode = 'test'
        else:
            mode = 'extra_val'
            extra_val_key = self.extra_val_titles[dataloader_idx - 2]
        self._is_training_step = False
        return self._do_step(batch, batch_idx, mode=mode, extra_val_key=extra_val_key)

    def training_step_end(self, batch_parts_outputs):
        if self.training and self.average_generator \
                and self.global_step >= self.average_generator_start_step \
                and self.global_step >= self.last_generator_averaging_step + self.average_generator_period:
            if self.generator_average is None:
                self.generator_average = copy.deepcopy(self.generator)
            else:
                update_running_average(self.generator_average, self.generator, decay=self.generator_avg_beta)
            self.last_generator_averaging_step = self.global_step

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss

    def validation_epoch_end(self, outputs):
        outputs = [step_out for out_group in outputs for step_out in out_group]
        averaged_logs = average_dicts(step_out['log_info'] for step_out in outputs)
        self.log_dict({k: v.mean() for k, v in averaged_logs.items()})

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        # standard validation
        val_evaluator_states = [s['val_evaluator_state'] for s in outputs if 'val_evaluator_state' in s]
        val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
        val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
        val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{val_evaluator_res_df}')

        for k, v in flatten_dict(val_evaluator_res).items():
            self.log(f'val_{k}', v)

        # standard visual test
        test_evaluator_states = [s['test_evaluator_state'] for s in outputs
                                 if 'test_evaluator_state' in s]
        test_evaluator_res = self.test_evaluator.evaluation_end(states=test_evaluator_states)
        test_evaluator_res_df = pd.DataFrame(test_evaluator_res).stack(1).unstack(0)
        test_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Test metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{test_evaluator_res_df}')

        for k, v in flatten_dict(test_evaluator_res).items():
            self.log(f'test_{k}', v)

        # extra validations
        if self.extra_evaluators:
            for cur_eval_title, cur_evaluator in self.extra_evaluators.items():
                cur_state_key = f'extra_val_{cur_eval_title}_evaluator_state'
                cur_states = [s[cur_state_key] for s in outputs if cur_state_key in s]
                cur_evaluator_res = cur_evaluator.evaluation_end(states=cur_states)
                cur_evaluator_res_df = pd.DataFrame(cur_evaluator_res).stack(1).unstack(0)
                cur_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
                LOGGER.info(f'Extra val {cur_eval_title} metrics after epoch #{self.current_epoch}, '
                            f'total {self.global_step} iterations:\n{cur_evaluator_res_df}')
                for k, v in flatten_dict(cur_evaluator_res).items():
                    self.log(f'extra_val_{cur_eval_title}_{k}', v)

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None, extra_val_key=None):
        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        batch = self(batch)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        if self.get_ddp_rank() in (None, 0) and (batch_idx % self.visualize_each_iters == 0 or mode == 'test'):
            if self.config.losses.adversarial.weight > 0:
                if self.store_discr_outputs_for_vis:
                    with torch.no_grad():
                        self.store_discr_outputs(batch)
            vis_suffix = f'_{mode}'
            if mode == 'extra_val':
                vis_suffix += f'_{extra_val_key}'
            self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)

        metrics_prefix = f'{mode}_'
        if mode == 'extra_val':
            metrics_prefix += f'{extra_val_key}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if mode == 'val':
            result['val_evaluator_state'] = self.val_evaluator.process_batch(batch)
        elif mode == 'test':
            result['test_evaluator_state'] = self.test_evaluator.process_batch(batch)
        elif mode == 'extra_val':
            result[f'extra_val_{extra_val_key}_evaluator_state'] = self.extra_evaluators[extra_val_key].process_batch(batch)

        return result

    def get_current_generator(self, no_average=False):
        if not no_average and not self.training and self.average_generator and self.generator_average is not None:
            return self.generator_average
        return self.generator

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pass data through generator and obtain at leas 'predicted_image' and 'inpainted' keys"""
        raise NotImplementedError()

    def generator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def store_discr_outputs(self, batch):
        out_size = batch['image'].shape[2:]
        discr_real_out, _ = self.discriminator(batch['image'])
        discr_fake_out, _ = self.discriminator(batch['predicted_image'])
        batch['discr_output_real'] = F.interpolate(discr_real_out, size=out_size, mode='nearest')
        batch['discr_output_fake'] = F.interpolate(discr_fake_out, size=out_size, mode='nearest')
        batch['discr_output_diff'] = batch['discr_output_real'] - batch['discr_output_fake']

    def get_ddp_rank(self):
        return self.trainer.global_rank if (self.trainer.num_nodes * self.trainer.num_processes) > 1 else None
