" Utils for modifying what is displayed in notebooks and command line"
import fastai
import fastprogress

from ..basic_train import *
from ..core import *

__all__ = ['progress_disabled_ctx']

class progress_disabled_ctx():
    "Context manager to disable the progress update bar and Recorder print."
    def __init__(self,learn:Learner):
        self.learn = learn

    def __enter__(self):
        #silence progress bar
        fastprogress.fastprogress.NO_BAR = True
        fastai.basic_train.master_bar,fastai.basic_train.progress_bar = fastprogress.force_console_behavior()
        self.orig_callback_fns = copy(self.learn.callback_fns)
        rec_name = [x for x in self.learn.callback_fns if hasattr(x, 'func') and x.func == Recorder]
        if len(rec_name):
            rec_idx = self.learn.callback_fns.index(rec_name[0])
            self.learn.callback_fns[rec_idx] = partial(Recorder, add_time=True, silent=True) #silence recorder
        return self.learn

    def __exit__(self, *args):
        fastai.basic_train.master_bar,fastai.basic_train.progress_bar = master_bar,progress_bar
        self.learn.callback_fns = self.orig_callback_fns
