from .torch_core import *
from .basic_train import Learner,LearnerCallback
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data.distributed import DistributedSampler

from fastai.text import TextLMDataBunch

__all__ = ['DistributedRecorder', 'DistributedTrainer', 'read_metrics', 'setup_distrib']

def rnn_reset(self):
    if hasattr(self.module, 'reset'): self.module.reset()
DistributedDataParallel.reset = rnn_reset

class ParallelTrainer(LearnerCallback):
    _order = -20
    def on_train_begin(self, **kwargs): self.learn.model = DataParallel(self.learn.model)
    def on_train_end  (self, **kwargs): self.learn.model = self.learn.model.module

class DistributedTrainer(LearnerCallback):
    _order = -20 # Needs to run before the recorder
    def __init__(self, learn:Learner, cuda_id:int=0):
        super().__init__(learn)
        self.cuda_id,self.train_sampler = cuda_id,None

    def _change_dl(self, dl, shuffle):
        old_dl = dl
        sampler = OurDistributedSampler(dl.dataset, shuffle=shuffle)
        new_dl = dl.new(shuffle=False, sampler=sampler)
        return old_dl,new_dl,sampler

    def on_train_begin(self, **kwargs):
        self.learn.model = DistributedDataParallel(self.model, device_ids=[self.cuda_id], output_device=self.cuda_id)
        shuffle = self.data.train_dl.init_kwargs['shuffle'] if hasattr(self.data.train_dl, 'init_kwargs') else True
        self.old_train_dl,self.data.train_dl,self.train_sampler = self._change_dl(self.data.train_dl, shuffle)
        if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
            self.old_valid_dl,self.data.valid_dl,self.valid_sampler = self._change_dl(self.data.valid_dl, shuffle)
        self.rank = rank_distrib()
        self.recorder.silent = (self.rank != 0)

    def on_epoch_begin(self, epoch, **kwargs): self.train_sampler.set_epoch(epoch)

    def on_train_end(self, **kwargs):
        self.learn.model = self.learn.model.module
        self.learn.data.train_dl = self.old_train_dl
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl = self.old_valid_dl

class DistributedRecorder(LearnerCallback):
    def __init__(self, learn:Learner, cuda_id:int=0, cache_dir:PathOrStr='tmp'):
        super().__init__(learn)
        self.cuda_id,self.cache_dir = cuda_id,cache_dir

    def on_train_begin(self, **kwargs):
        os.makedirs(self.learn.path/self.cache_dir, exist_ok=True)

    def on_epoch_end(self, **kwargs): self.save_stats()
    def on_train_end(self, **kwargs): self.save_stats()

    def save_stats(self):
        cache_path,recorder = self.learn.path/self.cache_dir,self.learn.recorder
        np.save(cache_path/f'losses_{self.cuda_id}', np.array(recorder.losses))
        stats = np.array([[v] + m for v,m in zip(recorder.val_losses,recorder.metrics)])
        np.save(cache_path/f'metrics_{self.cuda_id}', stats)

def _learner_parallel(learn:Learner):
    "Use nn.DataParallel when training and remove when done"
    if not torch.cuda.is_available(): warnings.warn('CUDA is not available, check your drivers - training will continue on CPU', ResourceWarning) 
    learn.callbacks.append(ParallelTrainer(learn))
    return learn

def _learner_distributed(learn:Learner, cuda_id:int, cache_dir:PathOrStr='tmp'):
    "Put `learn` on distributed training with `cuda_id`."
    learn.callbacks.append(DistributedTrainer(learn, cuda_id))
    learn.callbacks.append(DistributedRecorder(learn, cuda_id, cache_dir))
    return learn

Learner.to_distributed = _learner_distributed
Learner.to_parallel = _learner_parallel

def read_metrics(cache_path:PathOrStr, n_gpus:int, reduce:bool=True):
    losses,metrics = [],[]
    for i in range(n_gpus):
        losses.append(np.load(cache_path/f'losses_{i}.npy')[None])
        metrics.append(np.load(cache_path/f'metrics_{i}.npy')[None])
    if reduce:
        losses,metrics = np.concatenate(losses,0),np.concatenate(metrics,0)
        return losses.mean(0),metrics.mean(0)
    return losses,metrics

def setup_distrib(gpu:Any=None):
    if gpu is None: return gpu
    gpu = int(gpu)
    torch.cuda.set_device(int(gpu))
    if num_distrib() > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return gpu

class OurDistributedSampler(DistributedSampler):
    "A sampler for language models with the option to not shuffle."
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
            super().__init__(dataset, num_replicas=num_replicas, rank=rank)
            self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else: indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
