"Tools to help find the optimal learning rate for training"
from ..torch_core import *
from ..basic_data import DataBunch
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['LRFinder']

class LRFinder(LearnerCallback):
    "Causes `learn` to go on a mock training from `start_lr` to `end_lr` for `num_it` iterations."
    def __init__(self, learn:Learner, start_lr:float=1e-7, end_lr:float=10, num_it:int=100, stop_div:bool=True):
        super().__init__(learn)
        self.data,self.stop_div = learn.data,stop_div
        self.sched = Scheduler((start_lr, end_lr), num_it, annealing_exp)

    def on_train_begin(self, pbar, **kwargs:Any)->None:
        "Initialize optimizer and learner hyperparameters."
        setattr(pbar, 'clean_on_interrupt', True)
        self.learn.save('tmp')
        self.opt = self.learn.opt
        self.opt.lr = self.sched.start
        self.stop,self.best_loss = False,0.
        return {'skip_validate': True}

    def on_batch_end(self, iteration:int, smooth_loss:TensorOrNumber, **kwargs:Any)->None:
        "Determine if loss has runaway and we should stop."
        if iteration==0 or smooth_loss < self.best_loss: self.best_loss = smooth_loss
        self.opt.lr = self.sched.step()
        if self.sched.is_done or (self.stop_div and (smooth_loss > 4*self.best_loss or torch.isnan(smooth_loss))):
            #We use the smoothed loss to decide on the stopping since it's less shaky.
            return {'stop_epoch': True, 'stop_training': True}

    def on_train_end(self, **kwargs:Any)->None:
        "Cleanup learn model weights disturbed during LRFinder exploration."
        self.learn.load('tmp', purge=False)
        if hasattr(self.learn.model, 'reset'): self.learn.model.reset()
        for cb in self.callbacks:
            if hasattr(cb, 'reset'): cb.reset()
        print('LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.')
