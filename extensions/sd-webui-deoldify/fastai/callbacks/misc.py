" Miscellaneous callbacks "

from fastai.callback import Callback

class StopAfterNBatches(Callback):
    "Stop training after n batches of the first epoch."
    def __init__(self, n_batches:int=2):
        self.stop,self.n_batches = False,n_batches-1 # iteration starts from 0

    def on_batch_end(self, iteration, **kwargs):
        if iteration == self.n_batches:
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
