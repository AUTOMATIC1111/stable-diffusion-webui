"Callbacks provides extensibility to the `basic_train` loop. See `train` for examples of custom callbacks."
from .basic_data import *
from .torch_core import *
import torch.distributed as dist

__all__ = ['AverageMetric', 'Callback', 'CallbackHandler', 'OptimWrapper', 'SmoothenValue', 'Scheduler', 'annealing_cos', 'CallbackList',
           'annealing_exp', 'annealing_linear', 'annealing_no', 'annealing_poly']

class OptimWrapper():
    "Basic wrapper around `opt` to simplify hyper-parameters changes."
    def __init__(self, opt:optim.Optimizer, wd:Floats=0., true_wd:bool=False, bn_wd:bool=True):
        assert not isinstance(opt, OptimWrapper)
        self.opt,self.true_wd,self.bn_wd = opt,true_wd,bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func:Union[type,Callable], lr:Union[float,Tuple,List], layer_groups:ModuleList, wd:Floats=0., 
               true_wd:bool=False, bn_wd:bool=True)->optim.Optimizer:
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_params = split_no_wd_params(layer_groups)
        opt = opt_func([{'params': p, 'lr':0} for p in split_params])
        opt = cls(opt, wd=wd, true_wd=true_wd, bn_wd=bn_wd)
        opt.lr,opt.opt_func = listify(lr, layer_groups),opt_func
        return opt

    def new(self, layer_groups:Collection[nn.Module], split_no_wd:bool=True):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        res = self.create(opt_func, self.lr, layer_groups, wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd)
        res.mom,res.beta = self.mom,self.beta
        return res

    def new_with_params(self, param_groups:Collection[Collection[nn.Parameter]]):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        opt = opt_func([{'params': p, 'lr':0} for p in param_groups])
        opt = self.__class__(opt, wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd)
        opt.lr,opt.opt_func,opt.mom,opt.beta = self.lr,opt_func,self.mom,self.beta
        return opt

    def __repr__(self)->str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    #Pytorch optimizer methods
    def step(self)->None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr,wd,pg1,pg2 in zip(self._lr,self._wd,self.opt.param_groups[::2],self.opt.param_groups[1::2]):
                for p in pg1['params']: p.data.mul_(1 - wd*lr)
                if self.bn_wd:
                    for p in pg2['params']: p.data.mul_(1 - wd*lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self)->None:
        "Clear optimizer gradients."
        self.opt.zero_grad()

    #Passthrough to the inner opt.
    def __getattr__(self, k:str)->Any: return getattr(self.opt, k, None)
    def __setstate__(self,data:Any): self.__dict__.update(data)

    def clear(self):
        "Reset the state of the inner optimizer."
        sd = self.state_dict()
        sd['state'] = {}
        self.load_state_dict(sd)

    @property
    def n_params(self): return sum([len(pg['params']) for pg in self.opt.param_groups])

    #Hyperparameters as properties
    @property
    def lr(self)->float: return self._lr[-1]
    @lr.setter
    def lr(self, val:float)->None:
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self)->float:return self._mom[-1]
    @mom.setter
    def mom(self, val:float)->None:
        if 'momentum' in self.opt_keys: self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:  self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self)->float: return None if self._beta is None else self._beta[-1]
    @beta.setter
    def beta(self, val:float)->None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:    self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:  self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self)->float: return self._wd[-1]
    @wd.setter
    def wd(self, val:float)->None:
        "Set weight decay."
        if not self.true_wd: self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    #Helper functions
    def read_defaults(self)->None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom,self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')
        reserved_names = ['params', 'lr', 'momentum', 'alpha', 'betas', 'weight_decay']
        stat_names = [n for n in self.opt_keys if n not in reserved_names]
        self._stats = {n:self.read_val(n) for n in stat_names}

    def get_stat(self, name:str)->float: 
        if name in ['lr', 'mom', 'beta', 'wd']: return getattr(self, name)
        else: return self._stats[name][-1]
    def set_stat(self, name:str, value:Union[float, Collection[float]])->None:
        if name in ['lr', 'mom', 'beta', 'wd']: setattr(self, name, value)
        else:
            val = listify(value, self._stats[name])
            self.set_val(name, val)
            self._stats[name] = val

    def set_val(self, key:str, val:Any, bn_groups:bool=True)->Any:
        "Set `val` inside the optimizer dictionary at `key`."
        if is_tuple(val): val = [(v1,v2) for v1,v2 in zip(*val)]
        for v,pg1,pg2 in zip(val,self.opt.param_groups[::2],self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key:str) -> Union[List[float],Tuple[List[float],List[float]]]:
        "Read a hyperparameter `key` in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val
    
    def get_state(self):
        "Return the inner state minus the layer groups."
        return {'opt_state':self.opt.state_dict(), 'lr':self._lr, 'wd':self._wd, 'beta':self._beta, 'mom':self._mom,
                'opt_func':self.opt_func, 'true_wd':self.true_wd, 'bn_wd':self.bn_wd}

    @classmethod
    def load_with_state_and_layer_group(cls, state:dict, layer_groups:Collection[nn.Module]):
        res = cls.create(state['opt_func'], state['lr'], layer_groups, wd=state['wd'], true_wd=state['true_wd'], 
                     bn_wd=state['bn_wd'])
        res._mom,res._beta = state['mom'],state['beta']
        res.load_state_dict(state['opt_state'])
        return res

class Callback():
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order=0
    def on_train_begin(self, **kwargs:Any)->None:
        "To initialize constants in the callback."
        pass
    def on_epoch_begin(self, **kwargs:Any)->None:
        "At the beginning of each epoch."
        pass
    def on_batch_begin(self, **kwargs:Any)->None:
        "Set HP before the output and loss are computed."
        pass
    def on_loss_begin(self, **kwargs:Any)->None:
        "Called after forward pass but before loss has been computed."
        pass
    def on_backward_begin(self, **kwargs:Any)->None:
        "Called after the forward pass and the loss has been computed, but before backprop."
        pass
    def on_backward_end(self, **kwargs:Any)->None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass
    def on_step_end(self, **kwargs:Any)->None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass
    def on_batch_end(self, **kwargs:Any)->None:
        "Called at the end of the batch."
        pass
    def on_epoch_end(self, **kwargs:Any)->None:
        "Called at the end of an epoch."
        pass
    def on_train_end(self, **kwargs:Any)->None:
        "Useful for cleaning up things and saving files/models."
        pass
    def jump_to_epoch(self, epoch)->None:
        "To resume training at `epoch` directly."
        pass

    def get_state(self, minimal:bool=True):
        "Return the inner state of the `Callback`, `minimal` or not."
        to_remove = ['exclude', 'not_min'] + getattr(self, 'exclude', []).copy()
        if minimal: to_remove += getattr(self, 'not_min', []).copy()
        return {k:v for k,v in self.__dict__.items() if k not in to_remove}

    def  __repr__(self):
        attrs = func_args(self.__init__)
        to_remove = getattr(self, 'exclude', [])
        list_repr = [self.__class__.__name__] + [f'{k}: {getattr(self, k)}' for k in attrs if k != 'self' and k not in to_remove]
        return '\n'.join(list_repr)

class SmoothenValue():
    "Create a smooth moving average for a value (loss, etc) using `beta`."
    def __init__(self, beta:float):
        self.beta,self.n,self.mov_avg = beta,0,0

    def add_value(self, val:float)->None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)

CallbackList = Collection[Callback]

def _get_init_state(): return {'epoch':0, 'iteration':0, 'num_batch':0, 'skip_validate': False}

@dataclass
class CallbackHandler():
    "Manage all of the registered `callbacks` and `metrics`, smoothing loss by momentum `beta`."
    callbacks:CallbackList=None
    metrics:CallbackList=None
    beta:float=0.98

    def __post_init__(self)->None:
        "Initialize smoother and learning stats."
        self.callbacks = ifnone(self.callbacks, [])
        self.metrics = ifnone(self.metrics, [])
        self.metrics = [(met if isinstance(met, Callback) else AverageMetric(met)) for met in self.metrics]
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.smoothener = SmoothenValue(self.beta)
        self.state_dict:Dict[str,Union[int,float,Tensor]]=_get_init_state()

    def _call_and_update(self, cb, cb_name, **kwargs)->None:
        "Call `cb_name` on `cb` and update the inner state."
        new = ifnone(getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs), dict())
        for k,v in new.items():
            if k not in self.state_dict:
                raise Exception(f"{k} isn't a valid key in the state of the callbacks.")
            else: self.state_dict[k] = v
    
    def __call__(self, cb_name, call_mets=True, **kwargs)->None:
        "Call through to all of the `CallbakHandler` functions."
        if call_mets: 
            for met in self.metrics: self._call_and_update(met, cb_name, **kwargs)
        for cb in self.callbacks: self._call_and_update(cb, cb_name, **kwargs)

    def set_dl(self, dl:DataLoader):
        "Set the current `dl` used."
        if hasattr(self, 'cb_dl'): self.callbacks.remove(self.cb_dl)
        if isinstance(dl.dataset, Callback):
            self.callbacks.append(dl.dataset)
            self.cb_dl = dl.dataset

    def on_train_begin(self, epochs:int, pbar:PBar, metrics:MetricFuncList)->None:
        "About to start learning."
        self.state_dict = _get_init_state()
        self.state_dict.update(dict(n_epochs=epochs, pbar=pbar, metrics=metrics))
        names = [(met.name if hasattr(met, 'name') else camel2snake(met.__class__.__name__)) for met in self.metrics]
        self('train_begin', metrics_names=names)
        if self.state_dict['epoch'] != 0:
            self.state_dict['pbar'].first_bar.total -= self.state_dict['epoch']
            for cb in self.callbacks: cb.jump_to_epoch(self.state_dict['epoch'])

    def on_epoch_begin(self)->None:
        "Handle new epoch."
        self.state_dict['num_batch'],self.state_dict['stop_training'] = 0,False
        self('epoch_begin')

    def on_batch_begin(self, xb:Tensor, yb:Tensor, train:bool=True)->Tuple[Any,Any]:
        "Handle new batch `xb`,`yb` in `train` or validation."
        self.state_dict.update(dict(last_input=xb, last_target=yb, train=train, 
            stop_epoch=False, skip_step=False, skip_zero=False, skip_bwd=False))
        self('batch_begin', mets = not self.state_dict['train'])
        return self.state_dict['last_input'], self.state_dict['last_target']

    def on_loss_begin(self, out:Tensor)->Any:
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        self('loss_begin', call_mets=False)
        return self.state_dict['last_output']

    def on_backward_begin(self, loss:Tensor)->Tuple[Any,Any]:
        "Handle gradient calculation on `loss`."
        self.smoothener.add_value(loss.detach().cpu())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        self('backward_begin', call_mets=False)
        return self.state_dict['last_loss'], self.state_dict['skip_bwd']

    def on_backward_end(self)->Any:
        "Handle end of gradient calculation."
        self('backward_end', call_mets=False)
        return self.state_dict['skip_step']

    def on_step_end(self)->Any:
        "Handle end of optimization step."
        self('step_end', call_mets=False)
        return self.state_dict['skip_zero']

    def on_batch_end(self, loss:Tensor)->Any:
        "Handle end of processing one batch with `loss`."
        self.state_dict['last_loss'] = loss
        self('batch_end', call_mets = not self.state_dict['train'])
        if self.state_dict['train']:
            self.state_dict['iteration'] += 1
            self.state_dict['num_batch'] += 1
        return self.state_dict['stop_epoch']

    def on_epoch_end(self, val_loss:Tensor)->bool:
        "Epoch is done, process `val_loss`."
        self.state_dict['last_metrics'] = [val_loss] if val_loss is not None else [None]
        self('epoch_end', call_mets = val_loss is not None)
        self.state_dict['epoch'] += 1
        return self.state_dict['stop_training']

    def on_train_end(self, exception:Union[bool,Exception])->None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self('train_end', exception=exception)
        
    @property
    def skip_validate(self): return self.state_dict['skip_validate']

class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, func):
        # If func has a __name__ use this one else it should be a partial
        name = func.__name__ if hasattr(func, '__name__') else func.func.__name__
        self.func, self.name = func, name
        self.world = num_distrib()

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        self.count += first_el(last_target).size(0)
        val = self.func(last_output, *last_target)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)

def annealing_no(start:Number, end:Number, pct:float)->Number:
    "No annealing, always return `start`."
    return start
def annealing_linear(start:Number, end:Number, pct:float)->Number:
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)
def annealing_exp(start:Number, end:Number, pct:float)->Number:
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct
def annealing_cos(start:Number, end:Number, pct:float)->Number:
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

def do_annealing_poly(start:Number, end:Number, pct:float, degree:Number)->Number:
    "Helper function for `anneal_poly`."
    return end + (start-end) * (1-pct)**degree
def annealing_poly(degree:Number)->Number:
    "Anneal polynomically from `start` to `end` as pct goes from 0.0 to 1.0."
    return functools.partial(do_annealing_poly, degree=degree)

class Scheduler():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func`"
    def __init__(self, vals:StartOptEnd, n_iter:int, func:Optional[AnnealFunc]=None):
        self.start,self.end = (vals[0],vals[1]) if is_tuple(vals) else (vals,0)
        self.n_iter = max(1,n_iter)
        if func is None: self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:          self.func = func
        self.n = 0
        
    def restart(self): self.n = 0

    def step(self)->Number:
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)

    @property
    def is_done(self)->bool:
        "Return `True` if schedule completed."
        return self.n >= self.n_iter

