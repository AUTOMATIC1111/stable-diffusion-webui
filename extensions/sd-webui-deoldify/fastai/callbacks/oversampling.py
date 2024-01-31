from ..torch_core import *
from ..basic_data import DataBunch
from ..callback import *
from ..basic_train import Learner,LearnerCallback
from torch.utils.data.sampler import WeightedRandomSampler

__all__ = ['OverSamplingCallback']



class OverSamplingCallback(LearnerCallback):
    def __init__(self,learn:Learner,weights:torch.Tensor=None):
        super().__init__(learn)
        self.labels = self.learn.data.train_dl.dataset.y.items
        _, counts = np.unique(self.labels,return_counts=True)
        self.weights = (weights if weights is not None else
                        torch.DoubleTensor((1/counts)[self.labels]))
        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i].data for i in range(len(self.learn.data.train_dl.dataset))])
        self.total_len_oversample = int(self.learn.data.c*np.max(self.label_counts))
        
    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(self.weights,self.total_len_oversample), self.learn.data.train_dl.batch_size,False)