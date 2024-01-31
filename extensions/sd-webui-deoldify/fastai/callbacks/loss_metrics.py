from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['LossMetrics']

class LossMetrics(LearnerCallback):
    "Add `loss_func.metrics` to metrics named by `loss_func.metric_names`"
    _order = -20 #Needs to run before the recorder

    def on_train_begin(self, **kwargs):
        "Add the metrics names to the `Recorder`."
        self.names = ifnone(self.learn.loss_func.metric_names, [])
        if not self.names: warn('LossMetrics requested but no loss_func.metric_names provided')
        self.learn.recorder.add_metric_names(self.names)

    def on_epoch_begin(self, **kwargs):
        "Initialize the metrics for this epoch."
        self.metrics = {name:0. for name in self.names}
        self.nums = 0

    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        bs = last_target.size(0)
        for name in self.names:
            self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if not self.nums: return
        metrics = [self.metrics[name]/self.nums for name in self.names]
        return {'last_metrics': last_metrics+metrics}
