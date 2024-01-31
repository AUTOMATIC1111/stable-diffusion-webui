# Copyright (c) OpenMMLab. All rights reserved.
from ...parallel import is_module_wrapper
from ..hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class EMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook.

        .. math::

            \text{Xema\_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema\_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str): The checkpoint path. Defaults to None.
    """

    def __init__(self,
                 momentum=0.0002,
                 interval=1,
                 warm_up=100,
                 resume_from=None):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum**interval
        self.checkpoint = resume_from

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        # We warm up the momentum considering the instability at beginning
        momentum = min(self.momentum,
                       (1 + curr_step) / (self.warm_up + curr_step))
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(1 - momentum).add_(momentum, parameter.data)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
