# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, List, Optional, Union

import torch

from ..dist_utils import master_only
from .hook import HOOKS, Hook


@HOOKS.register_module()
class ProfilerHook(Hook):
    """Profiler to analyze performance during training.

    PyTorch Profiler is a tool that allows the collection of the performance
    metrics during the training. More details on Profiler can be found at
    https://pytorch.org/docs/1.8.1/profiler.html#torch.profiler.profile

    Args:
        by_epoch (bool): Profile performance by epoch or by iteration.
            Default: True.
        profile_iters (int): Number of iterations for profiling.
            If ``by_epoch=True``, profile_iters indicates that they are the
            first profile_iters epochs at the beginning of the
            training, otherwise it indicates the first profile_iters
            iterations. Default: 1.
        activities (list[str]): List of activity groups (CPU, CUDA) to use in
            profiling. Default: ['cpu', 'cuda'].
        schedule (dict, optional): Config of generating the callable schedule.
            if schedule is None, profiler will not add step markers into the
            trace and table view. Default: None.
        on_trace_ready (callable, dict): Either a handler or a dict of generate
            handler. Default: None.
        record_shapes (bool): Save information about operator's input shapes.
            Default: False.
        profile_memory (bool): Track tensor memory allocation/deallocation.
            Default: False.
        with_stack (bool): Record source information (file and line number)
            for the ops. Default: False.
        with_flops (bool): Use formula to estimate the FLOPS of specific
            operators (matrix multiplication and 2D convolution).
            Default: False.
        json_trace_path (str, optional): Exports the collected trace in Chrome
            JSON format. Default: None.

    Example:
        >>> runner = ... # instantiate a Runner
        >>> # tensorboard trace
        >>> trace_config = dict(type='tb_trace', dir_name='work_dir')
        >>> profiler_config = dict(on_trace_ready=trace_config)
        >>> runner.register_profiler_hook(profiler_config)
        >>> runner.run(data_loaders=[trainloader], workflow=[('train', 1)])
    """

    def __init__(self,
                 by_epoch: bool = True,
                 profile_iters: int = 1,
                 activities: List[str] = ['cpu', 'cuda'],
                 schedule: Optional[dict] = None,
                 on_trace_ready: Optional[Union[Callable, dict]] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:
        try:
            from torch import profiler  # torch version >= 1.8.1
        except ImportError:
            raise ImportError('profiler is the new feature of torch1.8.1, '
                              f'but your version is {torch.__version__}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean.'
        self.by_epoch = by_epoch

        if profile_iters < 1:
            raise ValueError('profile_iters should be greater than 0, but got '
                             f'{profile_iters}')
        self.profile_iters = profile_iters

        if not isinstance(activities, list):
            raise ValueError(
                f'activities should be list, but got {type(activities)}')
        self.activities = []
        for activity in activities:
            activity = activity.lower()
            if activity == 'cpu':
                self.activities.append(profiler.ProfilerActivity.CPU)
            elif activity == 'cuda':
                self.activities.append(profiler.ProfilerActivity.CUDA)
            else:
                raise ValueError(
                    f'activity should be "cpu" or "cuda", but got {activity}')

        if schedule is not None:
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.json_trace_path = json_trace_path

    @master_only
    def before_run(self, runner):
        if self.by_epoch and runner.max_epochs < self.profile_iters:
            raise ValueError('self.profile_iters should not be greater than '
                             f'{runner.max_epochs}')

        if not self.by_epoch and runner.max_iters < self.profile_iters:
            raise ValueError('self.profile_iters should not be greater than '
                             f'{runner.max_iters}')

        if callable(self.on_trace_ready):  # handler
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):  # config of handler
            trace_cfg = self.on_trace_ready.copy()
            trace_type = trace_cfg.pop('type')  # log_trace handler
            if trace_type == 'log_trace':

                def _log_handler(prof):
                    print(prof.key_averages().table(**trace_cfg))

                _on_trace_ready = _log_handler
            elif trace_type == 'tb_trace':  # tensorboard_trace handler
                try:
                    import torch_tb_profiler  # noqa: F401
                except ImportError:
                    raise ImportError('please run "pip install '
                                      'torch-tb-profiler" to install '
                                      'torch_tb_profiler')
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    **trace_cfg)
            else:
                raise ValueError('trace_type should be "log_trace" or '
                                 f'"tb_trace", but got {trace_type}')
        elif self.on_trace_ready is None:
            _on_trace_ready = None  # type: ignore
        else:
            raise ValueError('on_trace_ready should be handler, dict or None, '
                             f'but got {type(self.on_trace_ready)}')

        if runner.max_epochs > 1:
            warnings.warn(f'profiler will profile {runner.max_epochs} epochs '
                          'instead of 1 epoch. Since profiler will slow down '
                          'the training, it is recommended to train 1 epoch '
                          'with ProfilerHook and adjust your setting according'
                          ' to the profiler summary. During normal training '
                          '(epoch > 1), you may disable the ProfilerHook.')

        self.profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=_on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops)

        self.profiler.__enter__()
        runner.logger.info('profiler is profiling...')

    @master_only
    def after_train_epoch(self, runner):
        if self.by_epoch and runner.epoch == self.profile_iters - 1:
            runner.logger.info('profiler may take a few minutes...')
            self.profiler.__exit__(None, None, None)
            if self.json_trace_path is not None:
                self.profiler.export_chrome_trace(self.json_trace_path)

    @master_only
    def after_train_iter(self, runner):
        self.profiler.step()
        if not self.by_epoch and runner.iter == self.profile_iters - 1:
            runner.logger.info('profiler may take a few minutes...')
            self.profiler.__exit__(None, None, None)
            if self.json_trace_path is not None:
                self.profiler.export_chrome_trace(self.json_trace_path)
