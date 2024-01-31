# Copyright (c) OpenMMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class NeptuneLoggerHook(LoggerHook):
    """Class to log metrics to NeptuneAI.

    It requires `neptune-client` to be installed.

    Args:
        init_kwargs (dict): a dict contains the initialization keys as below:
            - project (str): Name of a project in a form of
                namespace/project_name. If None, the value of
                NEPTUNE_PROJECT environment variable will be taken.
            - api_token (str): User’s API token.
                If None, the value of NEPTUNE_API_TOKEN environment
                variable will be taken. Note: It is strongly recommended
                to use NEPTUNE_API_TOKEN environment variable rather than
                placing your API token in plain text in your source code.
            - name (str, optional, default is 'Untitled'): Editable name of
                the run. Name is displayed in the run's Details and in
                Runs table as a column.
            Check https://docs.neptune.ai/api-reference/neptune#init for
                more init arguments.
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging
        by_epoch (bool): Whether EpochBasedRunner is used.

    .. _NeptuneAI:
        https://docs.neptune.ai/you-should-know/logging-metadata
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 with_step=True,
                 by_epoch=True):

        super(NeptuneLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self.import_neptune()
        self.init_kwargs = init_kwargs
        self.with_step = with_step

    def import_neptune(self):
        try:
            import neptune.new as neptune
        except ImportError:
            raise ImportError(
                'Please run "pip install neptune-client" to install neptune')
        self.neptune = neptune
        self.run = None

    @master_only
    def before_run(self, runner):
        if self.init_kwargs:
            self.run = self.neptune.init(**self.init_kwargs)
        else:
            self.run = self.neptune.init()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            for tag_name, tag_value in tags.items():
                if self.with_step:
                    self.run[tag_name].log(
                        tag_value, step=self.get_iter(runner))
                else:
                    tags['global_step'] = self.get_iter(runner)
                    self.run[tag_name].log(tags)

    @master_only
    def after_run(self, runner):
        self.run.stop()
