# Copyright (c) OpenMMLab. All rights reserved.
import annotator.mmpkg.mmcv as mmcv
from .hook import HOOKS, Hook
from .lr_updater import annealing_cos, annealing_linear, format_param


class MomentumUpdaterHook(Hook):

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.9):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_momentum" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_momentum = []  # initial momentum for all param groups
        self.regular_momentum = [
        ]  # expected momentum if no warming up is performed

    def _set_momentum(self, runner, momentum_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, mom in zip(optim.param_groups,
                                            momentum_groups[k]):
                    if 'momentum' in param_group.keys():
                        param_group['momentum'] = mom
                    elif 'betas' in param_group.keys():
                        param_group['betas'] = (mom, param_group['betas'][1])
        else:
            for param_group, mom in zip(runner.optimizer.param_groups,
                                        momentum_groups):
                if 'momentum' in param_group.keys():
                    param_group['momentum'] = mom
                elif 'betas' in param_group.keys():
                    param_group['betas'] = (mom, param_group['betas'][1])

    def get_momentum(self, runner, base_momentum):
        raise NotImplementedError

    def get_regular_momentum(self, runner):
        if isinstance(runner.optimizer, dict):
            momentum_groups = {}
            for k in runner.optimizer.keys():
                _momentum_group = [
                    self.get_momentum(runner, _base_momentum)
                    for _base_momentum in self.base_momentum[k]
                ]
                momentum_groups.update({k: _momentum_group})
            return momentum_groups
        else:
            return [
                self.get_momentum(runner, _base_momentum)
                for _base_momentum in self.base_momentum
            ]

    def get_warmup_momentum(self, cur_iters):

        def _get_warmup_momentum(cur_iters, regular_momentum):
            if self.warmup == 'constant':
                warmup_momentum = [
                    _momentum / self.warmup_ratio
                    for _momentum in self.regular_momentum
                ]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_momentum = [
                    _momentum / (1 - k) for _momentum in self.regular_mom
                ]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_momentum = [
                    _momentum / k for _momentum in self.regular_mom
                ]
            return warmup_momentum

        if isinstance(self.regular_momentum, dict):
            momentum_groups = {}
            for key, regular_momentum in self.regular_momentum.items():
                momentum_groups[key] = _get_warmup_momentum(
                    cur_iters, regular_momentum)
            return momentum_groups
        else:
            return _get_warmup_momentum(cur_iters, self.regular_momentum)

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint,
        # if 'initial_momentum' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_momentum = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    if 'momentum' in group.keys():
                        group.setdefault('initial_momentum', group['momentum'])
                    else:
                        group.setdefault('initial_momentum', group['betas'][0])
                _base_momentum = [
                    group['initial_momentum'] for group in optim.param_groups
                ]
                self.base_momentum.update({k: _base_momentum})
        else:
            for group in runner.optimizer.param_groups:
                if 'momentum' in group.keys():
                    group.setdefault('initial_momentum', group['momentum'])
                else:
                    group.setdefault('initial_momentum', group['betas'][0])
            self.base_momentum = [
                group['initial_momentum']
                for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_mom = self.get_regular_momentum(runner)
        self._set_momentum(runner, self.regular_mom)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_mom = self.get_regular_momentum(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_momentum(runner, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_momentum(runner, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)


@HOOKS.register_module()
class StepMomentumUpdaterHook(MomentumUpdaterHook):
    """Step momentum scheduler with min value clipping.

    Args:
        step (int | list[int]): Step to decay the momentum. If an int value is
            given, regard it as the decay interval. If a list is given, decay
            momentum at these steps.
        gamma (float, optional): Decay momentum ratio. Default: 0.5.
        min_momentum (float, optional): Minimum momentum value to keep. If
            momentum after decay is lower than this value, it will be clipped
            accordingly. If None is given, we don't perform lr clipping.
            Default: None.
    """

    def __init__(self, step, gamma=0.5, min_momentum=None, **kwargs):
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_momentum = min_momentum
        super(StepMomentumUpdaterHook, self).__init__(**kwargs)

    def get_momentum(self, runner, base_momentum):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        momentum = base_momentum * (self.gamma**exp)
        if self.min_momentum is not None:
            # clip to a minimum value
            momentum = max(momentum, self.min_momentum)
        return momentum


@HOOKS.register_module()
class CosineAnnealingMomentumUpdaterHook(MomentumUpdaterHook):

    def __init__(self, min_momentum=None, min_momentum_ratio=None, **kwargs):
        assert (min_momentum is None) ^ (min_momentum_ratio is None)
        self.min_momentum = min_momentum
        self.min_momentum_ratio = min_momentum_ratio
        super(CosineAnnealingMomentumUpdaterHook, self).__init__(**kwargs)

    def get_momentum(self, runner, base_momentum):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        if self.min_momentum_ratio is not None:
            target_momentum = base_momentum * self.min_momentum_ratio
        else:
            target_momentum = self.min_momentum
        return annealing_cos(base_momentum, target_momentum,
                             progress / max_progress)


@HOOKS.register_module()
class CyclicMomentumUpdaterHook(MomentumUpdaterHook):
    """Cyclic momentum Scheduler.

    Implement the cyclical momentum scheduler policy described in
    https://arxiv.org/pdf/1708.07120.pdf

    This momentum scheduler usually used together with the CyclicLRUpdater
    to improve the performance in the 3D detection area.

    Attributes:
        target_ratio (tuple[float]): Relative ratio of the lowest momentum and
            the highest momentum to the initial momentum.
        cyclic_times (int): Number of cycles during training
        step_ratio_up (float): The ratio of the increasing process of momentum
            in  the total cycle.
        by_epoch (bool): Whether to update momentum by epoch.
    """

    def __init__(self,
                 by_epoch=False,
                 target_ratio=(0.85 / 0.95, 1),
                 cyclic_times=1,
                 step_ratio_up=0.4,
                 **kwargs):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple, got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.momentum_phases = []  # init momentum_phases
        # currently only support by_epoch=False
        assert not by_epoch, \
            'currently only support "by_epoch" = False'
        super(CyclicMomentumUpdaterHook, self).__init__(by_epoch, **kwargs)

    def before_run(self, runner):
        super(CyclicMomentumUpdaterHook, self).before_run(runner)
        # initiate momentum_phases
        # total momentum_phases are separated as up and down
        max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.momentum_phases.append(
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.momentum_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])

    def get_momentum(self, runner, base_momentum):
        curr_iter = runner.iter
        for (start_iter, end_iter, max_iter_per_phase, start_ratio,
             end_ratio) in self.momentum_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                return annealing_cos(base_momentum * start_ratio,
                                     base_momentum * end_ratio,
                                     progress / (end_iter - start_iter))


@HOOKS.register_module()
class OneCycleMomentumUpdaterHook(MomentumUpdaterHook):
    """OneCycle momentum Scheduler.

    This momentum scheduler usually used together with the OneCycleLrUpdater
    to improve the performance.

    Args:
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is
            'max_momentum' and learning rate is 'base_lr'
            Default: 0.95
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
    """

    def __init__(self,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 three_phase=False,
                 **kwargs):
        # validate by_epoch, currently only support by_epoch=False
        if 'by_epoch' not in kwargs:
            kwargs['by_epoch'] = False
        else:
            assert not kwargs['by_epoch'], \
                'currently only support "by_epoch" = False'
        if not isinstance(base_momentum, (float, list, dict)):
            raise ValueError('base_momentum must be the type among of float,'
                             'list or dict.')
        self._base_momentum = base_momentum
        if not isinstance(max_momentum, (float, list, dict)):
            raise ValueError('max_momentum must be the type among of float,'
                             'list or dict.')
        self._max_momentum = max_momentum
        # validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError('Expected float between 0 and 1 pct_start, but '
                             f'got {pct_start}')
        self.pct_start = pct_start
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must by one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear
        self.three_phase = three_phase
        self.momentum_phases = []  # init momentum_phases
        super(OneCycleMomentumUpdaterHook, self).__init__(**kwargs)

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                if ('momentum' not in optim.defaults
                        and 'betas' not in optim.defaults):
                    raise ValueError('optimizer must support momentum with'
                                     'option enabled')
                self.use_beta1 = 'betas' in optim.defaults
                _base_momentum = format_param(k, optim, self._base_momentum)
                _max_momentum = format_param(k, optim, self._max_momentum)
                for group, b_momentum, m_momentum in zip(
                        optim.param_groups, _base_momentum, _max_momentum):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['base_momentum'] = b_momentum
                    group['max_momentum'] = m_momentum
        else:
            optim = runner.optimizer
            if ('momentum' not in optim.defaults
                    and 'betas' not in optim.defaults):
                raise ValueError('optimizer must support momentum with'
                                 'option enabled')
            self.use_beta1 = 'betas' in optim.defaults
            k = type(optim).__name__
            _base_momentum = format_param(k, optim, self._base_momentum)
            _max_momentum = format_param(k, optim, self._max_momentum)
            for group, b_momentum, m_momentum in zip(optim.param_groups,
                                                     _base_momentum,
                                                     _max_momentum):
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (m_momentum, beta2)
                else:
                    group['momentum'] = m_momentum
                group['base_momentum'] = b_momentum
                group['max_momentum'] = m_momentum

        if self.three_phase:
            self.momentum_phases.append({
                'end_iter':
                float(self.pct_start * runner.max_iters) - 1,
                'start_momentum':
                'max_momentum',
                'end_momentum':
                'base_momentum'
            })
            self.momentum_phases.append({
                'end_iter':
                float(2 * self.pct_start * runner.max_iters) - 2,
                'start_momentum':
                'base_momentum',
                'end_momentum':
                'max_momentum'
            })
            self.momentum_phases.append({
                'end_iter': runner.max_iters - 1,
                'start_momentum': 'max_momentum',
                'end_momentum': 'max_momentum'
            })
        else:
            self.momentum_phases.append({
                'end_iter':
                float(self.pct_start * runner.max_iters) - 1,
                'start_momentum':
                'max_momentum',
                'end_momentum':
                'base_momentum'
            })
            self.momentum_phases.append({
                'end_iter': runner.max_iters - 1,
                'start_momentum': 'base_momentum',
                'end_momentum': 'max_momentum'
            })

    def _set_momentum(self, runner, momentum_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, mom in zip(optim.param_groups,
                                            momentum_groups[k]):
                    if 'momentum' in param_group.keys():
                        param_group['momentum'] = mom
                    elif 'betas' in param_group.keys():
                        param_group['betas'] = (mom, param_group['betas'][1])
        else:
            for param_group, mom in zip(runner.optimizer.param_groups,
                                        momentum_groups):
                if 'momentum' in param_group.keys():
                    param_group['momentum'] = mom
                elif 'betas' in param_group.keys():
                    param_group['betas'] = (mom, param_group['betas'][1])

    def get_momentum(self, runner, param_group):
        curr_iter = runner.iter
        start_iter = 0
        for i, phase in enumerate(self.momentum_phases):
            end_iter = phase['end_iter']
            if curr_iter <= end_iter or i == len(self.momentum_phases) - 1:
                pct = (curr_iter - start_iter) / (end_iter - start_iter)
                momentum = self.anneal_func(
                    param_group[phase['start_momentum']],
                    param_group[phase['end_momentum']], pct)
                break
            start_iter = end_iter
        return momentum

    def get_regular_momentum(self, runner):
        if isinstance(runner.optimizer, dict):
            momentum_groups = {}
            for k, optim in runner.optimizer.items():
                _momentum_group = [
                    self.get_momentum(runner, param_group)
                    for param_group in optim.param_groups
                ]
                momentum_groups.update({k: _momentum_group})
            return momentum_groups
        else:
            momentum_groups = []
            for param_group in runner.optimizer.param_groups:
                momentum_groups.append(self.get_momentum(runner, param_group))
            return momentum_groups
