"Provides basic training and validation with `Learner`"
from .torch_core import *
from .basic_data import *
from .callback import *
from .data_block import *
from .utils.ipython import gpu_mem_restore
import inspect
from fastprogress.fastprogress import format_time, IN_NOTEBOOK
from time import time
from fastai.sixel import plot_sixel

__all__ = ['Learner', 'LearnerCallback', 'Recorder', 'RecordOnCPU', 'fit', 'loss_batch', 'train_epoch', 'validate',
           'get_preds', 'load_learner']

defaults.lr = slice(3e-3)
defaults.wd = 1e-2
defaults.extra_callbacks    = None
defaults.extra_callback_fns = None

def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None, count:[int]=[1], batch_multiplier:int=1)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = model(*xb)

    if not loss_func: return to_detach(out), yb[0].detach()
    out = cb_handler.on_loss_begin(out)
    loss = loss_func(out, *yb)/batch_multiplier
    count[0]-=1

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if count[0] == 0:
            if not cb_handler.on_backward_end(): opt.step()
            if not cb_handler.on_step_end():     opt.zero_grad()
            count[0] = batch_multiplier

    return loss.detach().cpu()

def get_preds(model:nn.Module, dl:DataLoader, pbar:Optional[PBar]=None, cb_handler:Optional[CallbackHandler]=None,
              activ:nn.Module=None, loss_func:OptLossFunc=None, n_batch:Optional[int]=None) -> List[Tensor]:
    "Tuple of predictions and targets, and optional losses (if `loss_func`) using `dl`, max batches `n_batch`."
    res = [torch.cat(o).cpu() for o in
           zip(*validate(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch))]
    if loss_func is not None:
        with NoneReduceOnCPU(loss_func) as lf: res.append(lf(res[0], res[1]))
    if activ is not None: res[0] = activ(res[0])
    return res

def validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             pbar:Optional[PBar]=None, average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        if cb_handler: cb_handler.set_dl(dl)
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler)
            val_losses.append(val_loss)
            if not is_listy(yb): yb = [yb]
            nums.append(first_el(yb).shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses

def train_epoch(model:nn.Module, dl:DataLoader, opt:optim.Optimizer, loss_func:LossFunction)->None:
    "Simple training of `model` for 1 epoch of `dl` using optim `opt` and loss function `loss_func`."
    model.train()
    for xb,yb in dl:
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

@dataclass
class BasicLearner():
    model:nn.Module
    loss_func:LossFunction
    opt:optim.Optimizer
    data:DataBunch

def fit(epochs:int, learn:BasicLearner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None, batch_multiplier:int=1)->None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
    assert len(learn.data.train_dl) != 0, f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception=False
    try:
        for epoch in pbar:
            learn.model.train()
            cb_handler.set_dl(learn.data.train_dl)
            cb_handler.on_epoch_begin()
            count = [batch_multiplier]
            for xb,yb in progress_bar(learn.data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(learn.model, xb, yb, learn.loss_func, learn.opt, cb_handler, count=count, batch_multiplier=batch_multiplier)
                if cb_handler.on_batch_end(loss): break

            if not cb_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(learn.model, learn.data.valid_dl, loss_func=learn.loss_func,
                                       cb_handler=cb_handler, pbar=pbar)
            else: val_loss=None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)

loss_func_name2activ = {'cross_entropy_loss': F.softmax, 'nll_loss': torch.exp, 'poisson_nll_loss': torch.exp,
    'kl_div_loss': torch.exp, 'bce_with_logits_loss': torch.sigmoid, 'cross_entropy': F.softmax,
    'kl_div': torch.exp, 'binary_cross_entropy_with_logits': torch.sigmoid,
}

def _loss_func_name2activ(name:str, axis:int=-1):
    res = loss_func_name2activ[name]
    if res == F.softmax: res = partial(F.softmax, dim=axis)
    return res

def _loss_func2activ(loss_func):
    if getattr(loss_func,'keywords',None):
        if not loss_func.keywords.get('log_input', True): return
    axis = getattr(loss_func, 'axis', -1)
    # flattened loss
    loss_func = getattr(loss_func, 'func', loss_func)
    # could have a partial inside flattened loss! Duplicate on purpose.
    loss_func = getattr(loss_func, 'func', loss_func)
    cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name == 'mix_up_loss':
        loss_func = loss_func.crit
        cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name in loss_func_name2activ:
        if cls_name == 'poisson_nll_loss' and (not getattr(loss_func, 'log_input', True)): return
        return _loss_func_name2activ(cls_name, axis)
    if getattr(loss_func,'__name__','') in loss_func_name2activ:
        return _loss_func_name2activ(loss_func.__name__, axis)
    return noop

@dataclass
class Learner():
    "Trainer for `model` using `data` to minimize `loss_func` with optimizer `opt_func`."
    data:DataBunch
    model:nn.Module
    opt_func:Callable=AdamW
    loss_func:Callable=None
    metrics:Collection[Callable]=None
    true_wd:bool=True
    bn_wd:bool=True
    wd:Floats=defaults.wd
    train_bn:bool=True
    path:str = None
    model_dir:PathOrStr = 'deoldify'
    callback_fns:Collection[Callable]=None
    callbacks:Collection[Callback]=field(default_factory=list)
    layer_groups:Collection[nn.Module]=None
    add_time:bool=True
    silent:bool=None
    def __post_init__(self)->None:
        "Setup path,metrics, callbacks and ensure model directory exists."
        self.path = Path(ifnone(self.path, self.data.path))
        self.model = self.model.to(self.data.device)
        self.loss_func = self.loss_func or self.data.loss_func
        self.metrics=listify(self.metrics)
        if not self.layer_groups: self.layer_groups = [nn.Sequential(*flatten_model(self.model))]
        self.callbacks = listify(self.callbacks)
        if self.silent is None: self.silent = defaults.silent
        self.callback_fns = [partial(Recorder, add_time=self.add_time, silent=self.silent)] + listify(self.callback_fns)

    def init(self, init): apply_init(self.model, init)

    def _test_writeable_path(self):
        path = self.path/self.model_dir
        try:
            path.mkdir(parents=True, exist_ok=True)
            tmp_file = get_tmp_file(path)
        except OSError as e:
            raise Exception(f"{e}\nCan't write to '{path}', set `learn.model_dir` attribute in Learner to a full libpath path that is writable") from None
        os.remove(tmp_file)

    def lr_range(self, lr:Union[float,slice])->np.ndarray:
        "Build differential learning rates from `lr`."
        if not isinstance(lr,slice): return lr
        if lr.start: res = even_mults(lr.start, lr.stop, len(self.layer_groups))
        else: res = [lr.stop/10]*(len(self.layer_groups)-1) + [lr.stop]
        return np.array(res)

    def fit(self, epochs:int, lr:Union[Floats,slice]=defaults.lr,
            wd:Floats=None, callbacks:Collection[Callback]=None, batch_multiplier:int=1)->None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        if not getattr(self, 'opt', False): self.create_opt(lr, wd)
        else: self.opt.lr,self.opt.wd = lr,wd
        callbacks = [cb(self) for cb in self.callback_fns + listify(defaults.extra_callback_fns)] + listify(callbacks)
        if defaults.extra_callbacks is not None: callbacks += defaults.extra_callbacks
        fit(epochs, self, metrics=self.metrics, callbacks=self.callbacks+callbacks, batch_multiplier=batch_multiplier)

    def create_opt(self, lr:Floats, wd:Floats=0.)->None:
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = OptimWrapper.create(self.opt_func, lr, self.layer_groups, wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)
        return self

    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)
        self.create_opt(defaults.lr)

    def freeze(self)->None:
        "Freeze up to last layer group."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def export(self, file:PathLikeOrBinaryStream='export.pkl', destroy=False):
        "Export the state of the `Learner` in `self.path/file`. `file` can be file-like (file or buffer)"
        if rank_distrib(): return # don't save if slave proc
        args = ['opt_func', 'loss_func', 'metrics', 'true_wd', 'bn_wd', 'wd', 'train_bn', 'model_dir', 'callback_fns']
        state = {a:getattr(self,a) for a in args}
        state['cb_state'] = {cb.__class__:cb.get_state() for cb in self.callbacks}
        #layer_groups -> need to find a way
        #TO SEE: do we save model structure and weights separately?
        with ModelOnCPU(self.model) as m:
            state['model'] = m
            xtra = dict(normalize=self.data.norm.keywords) if getattr(self.data, 'norm', False) else {}
            state['data'] = self.data.valid_ds.get_state(**xtra)
            state['cls'] = self.__class__
            try_save(state, self.path, file)
        if destroy: self.destroy()

    def save(self, file:PathLikeOrBinaryStream=None, return_path:bool=False, with_opt:bool=True):
        "Save model and optimizer state (if `with_opt`) with `file` to `self.model_dir`. `file` can be file-like (file or buffer)"
        if is_pathlike(file): self._test_writeable_path()
        if rank_distrib(): return # don't save if slave proc
        target = self.path/self.model_dir/f'{file}.pth' if is_pathlike(file) else file
        if not hasattr(self, 'opt'): with_opt=False
        if not with_opt: state = get_model(self.model).state_dict()
        else: state = {'model': get_model(self.model).state_dict(), 'opt':self.opt.state_dict()}
        torch.save(state, target)
        if return_path: return target

    def dl(self, ds_type:DatasetType=DatasetType.Valid):
        "Return DataLoader for DatasetType `ds_type`."
        return self.data.dl(ds_type)

    def load(self, file:PathLikeOrBinaryStream=None, device:torch.device=None, strict:bool=True,
             with_opt:bool=None, purge:bool=True, remove_module:bool=False):
        "Load model and optimizer state (if `with_opt`) `file` from `self.model_dir` using `device`. `file` can be file-like (file or buffer)"
        if purge: self.purge(clear_opt=ifnone(with_opt, False))
        if device is None: device = self.data.device
        elif isinstance(device, int): device = torch.device('cuda', device)
        source = self.path/self.model_dir/f'{file}.pth' if is_pathlike(file) else file
        state = torch.load(source, map_location=device)
        if set(state.keys()) == {'model', 'opt'}:
            model_state = state['model']
            if remove_module: model_state = remove_module_load(model_state)
            get_model(self.model).load_state_dict(model_state, strict=strict)
            if ifnone(with_opt,True):
                if not hasattr(self, 'opt'): self.create_opt(defaults.lr, self.wd)
                try:    self.opt.load_state_dict(state['opt'])
                except: pass
        else:
            if with_opt: warn("Saved filed doesn't contain an optimizer state.")
            if remove_module: state = remove_module_load(state)
            get_model(self.model).load_state_dict(state, strict=strict)
        del state
        gc.collect()
        return self

    def destroy(self):
        "Free the Learner internals, leaving just an empty shell that consumes no memory"

        class ZombieLearner(Learner):
            msg = "this object has been destroyed"
            def __getattr__(self, item):    print(ZombieLearner.msg); return None
            def destroyed(*args, **kwargs): print(ZombieLearner.msg)

        attrs = [k for k in self.__dict__.keys() if not k.startswith("__")]
        for a in attrs: delattr(self, a)
        # the instance methods can still be called, but will just give a message
        methods = [k for k in dir(self) if not k.startswith("__") and inspect.isroutine(getattr(self, k))]
        for m in methods: setattr(self, m, ZombieLearner.destroyed)
        self.__class__ = ZombieLearner
        gc.collect()
        print("this Learner object self-destroyed - it still exists, but no longer usable")

    def purge(self, clear_opt:bool=True):
        "Purge the `Learner` of all cached attributes to release some GPU memory."
        self._test_writeable_path()
        attrs_all = [k for k in self.__dict__.keys() if not k.startswith("__")]
        attrs_pkl = ['bn_wd', 'callback_fns', 'layer_groups', 'loss_func', 'metrics', 'model',
                     'model_dir', 'opt_func', 'path', 'train_bn', 'true_wd', 'wd']
        # +callbacks: get pickled too, but not directly
        attrs_keep = ['data', 'recorder']
        attrs_del = list(set(attrs_all) - set(attrs_keep))
        state = {a:getattr(self, a) for a in attrs_pkl}
        state['cb_state'] = {cb.__class__:cb.get_state() for cb in self.callbacks}
        if hasattr(self, 'opt'): state['opt'] = self.opt.get_state()

        tmp_file = get_tmp_file(self.path/self.model_dir)
        torch.save(state, open(tmp_file, 'wb'))
        for a in attrs_del: delattr(self, a)
        gc.collect()
        state = torch.load(tmp_file)
        os.remove(tmp_file)

        for a in attrs_pkl: setattr(self, a, state[a])
        cb_state = state.pop('cb_state')
        self.callbacks = [load_callback(c,s, self) for c,s in cb_state.items()]
        if not clear_opt and 'opt' in state:
            try: self.opt = OptimWrapper.load_with_state_and_layer_group(state['opt'], self.layer_groups)
            except: warn("Wasn't able to properly load the optimizer state again.")
        del state
        gc.collect()
        return self

    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None,
                  pbar:Optional[PBar]=None) -> List[Tensor]:
        "Return predictions and targets on `ds_type` dataset."
        lf = self.loss_func if with_loss else None
        return get_preds(self.model, self.dl(ds_type), cb_handler=CallbackHandler(self.callbacks),
                         activ=_loss_func2activ(self.loss_func), loss_func=lf, n_batch=n_batch, pbar=pbar)

    def pred_batch(self, ds_type:DatasetType=DatasetType.Valid, batch:Tuple=None, reconstruct:bool=False, with_dropout:bool=False) -> List[Tensor]:
        with torch.no_grad():
            training = self.model.training
            self.model.train(False)
            "Return output of the model on one batch from `ds_type` dataset."
            if batch is not None: xb,yb = batch
            else: xb,yb = self.data.one_batch(ds_type, detach=False, denorm=False)
            cb_handler = CallbackHandler(self.callbacks)
            xb,yb = cb_handler.on_batch_begin(xb,yb, train=False)
            if not with_dropout: 
                preds = loss_batch(self.model.eval(), xb, yb, cb_handler=cb_handler)
            else: 
                preds = loss_batch(self.model.eval().apply(self.apply_dropout), xb, yb, cb_handler=cb_handler)
            res = _loss_func2activ(self.loss_func)(preds[0])
            self.model.train(training)
            if not reconstruct: return res
            res = res.detach().cpu()
            ds = self.dl(ds_type).dataset
            norm = getattr(self.data, 'norm', False)
            if norm and norm.keywords.get('do_y',False):
                res = self.data.denorm(res, do_x=True)
            return [ds.reconstruct(o) for o in res]

    def backward(self, item):
        "Pass `item` through the model and computes the gradient. Useful if `backward_hooks` are attached."
        xb,yb = self.data.one_item(item)
        loss = loss_batch(self.model.eval(), xb, yb, self.loss_func, opt=FakeOptimizer(),
                          cb_handler=CallbackHandler(self.callbacks))
        return loss

    def predict(self, item:ItemBase, return_x:bool=False, batch_first:bool=True, with_dropout:bool=False, **kwargs):
        "Return predicted class, label and probabilities for `item`."
        batch = self.data.one_item(item)
        res = self.pred_batch(batch=batch, with_dropout=with_dropout)
        raw_pred,x = grab_idx(res,0,batch_first=batch_first),batch[0]
        norm = getattr(self.data,'norm',False)
        if norm:
            x = self.data.denorm(x)
            if norm.keywords.get('do_y',False): raw_pred = self.data.denorm(raw_pred)
        ds = self.data.single_ds
        pred = ds.y.analyze_pred(raw_pred, **kwargs)
        x = ds.x.reconstruct(grab_idx(x, 0))
        y = ds.y.reconstruct(pred, x) if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred)
        return (x, y, pred, raw_pred) if return_x else (y, pred, raw_pred)

    def validate(self, dl=None, callbacks=None, metrics=None):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.data.valid_dl)
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics)
        cb_handler.on_epoch_begin()
        val_metrics = validate(self.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        return cb_handler.state_dict['last_metrics']

    def show_results(self, ds_type=DatasetType.Valid, rows:int=5, **kwargs):
        "Show `rows` result of predictions on `ds_type` dataset."
        #TODO: get read of has_arg x and split_kwargs_by_func if possible
        #TODO: simplify this and refactor with pred_batch(...reconstruct=True)
        n_items = rows ** 2 if self.data.train_ds.x._square_show_res else rows
        if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
        ds = self.dl(ds_type).dataset
        self.callbacks.append(RecordOnCPU())
        preds = self.pred_batch(ds_type)
        *self.callbacks,rec_cpu = self.callbacks
        x,y = rec_cpu.input,rec_cpu.target
        norm = getattr(self.data,'norm',False)
        if norm:
            x = self.data.denorm(x)
            if norm.keywords.get('do_y',False):
                y     = self.data.denorm(y, do_x=True)
                preds = self.data.denorm(preds, do_x=True)
        analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(n_items)]
        xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        if has_arg(ds.y.reconstruct, 'x'):
            ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
            zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
        else :
            ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
            zs = [ds.y.reconstruct(z) for z in preds]
        ds.x.show_xyzs(xs, ys, zs, **kwargs)

    def apply_dropout(self, m):
        "If a module contains 'dropout' in it's name, it will be switched to .train() mode."
        if 'dropout' in m.__class__.__name__.lower(): m.train()

    def predict_with_mc_dropout(self, item:ItemBase, with_dropout:bool=True, n_times=10, **kwargs):
        "Make predictions with dropout turned on for n_times (default 10)."
        return [self.predict(item, with_dropout=with_dropout) for _ in range(n_times)]

class RecordOnCPU(Callback):
    "Store the `input` and `target` going through the model on the CPU."
    def on_batch_begin(self, last_input,last_target,**kwargs):
        self.input,self.target = to_cpu(last_input),to_cpu(last_target)

class LearnerCallback(Callback):
    "Base class for creating callbacks for a `Learner`."
    def __init__(self, learn):
        self._learn = weakref.ref(learn)
        self.exclude,self.not_min = ['_learn'],[]
        setattr(self.learn, self.cb_name, self)

    def __getattr__(self,k): return getattr(self.learn, k)
    def __setstate__(self,data:Any): self.__dict__.update(data)

    @property
    def learn(self) -> Learner: return self._learn()
    @learn.setter
    def learn(self, learn: Learner) -> None: self._learn = weakref.ref(learn)

    @property
    def cb_name(self): return camel2snake(self.__class__.__name__)

class Recorder(LearnerCallback):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."
    _order=-10
    def __init__(self, learn:Learner, add_time:bool=True, silent:bool=False):
        super().__init__(learn)
        self.opt = self.learn.opt
        self.train_dl = self.learn.data.train_dl
        self.no_val,self.silent,self.add_time = False,silent,add_time

    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        "Initialize recording status at beginning of training."
        self.pbar = pbar
        self.names = ['epoch', 'train_loss'] if self.no_val else ['epoch', 'train_loss', 'valid_loss']
        self.metrics_names = metrics_names
        if hasattr(self, '_added_met_names'): self.metrics_names += self._added_met_names
        self.names += self.metrics_names
        if self.add_time: self.names.append('time')
        if not self.silent: self.pbar.write(self.names, table=True)
        self.losses,self.val_losses,self.lrs,self.moms,self.metrics,self.nb_batches = [],[],[],[],[],[]

    def on_epoch_begin(self, **kwargs:Any)->None:
        if self.add_time: self.start_epoch = time()

    def on_batch_begin(self, train, **kwargs:Any)->None:
        "Record learning rate and momentum at beginning of batch."
        if train:
            self.lrs.append(self.opt.lr)
            self.moms.append(self.opt.mom)

    def on_backward_begin(self, smooth_loss:Tensor, **kwargs:Any)->None:
        "Record the loss before any other callback has a chance to modify it."
        self.losses.append(smooth_loss)
        if self.pbar is not None and hasattr(self.pbar,'child'):
            self.pbar.child.comment = f'{smooth_loss:.4f}'

    def on_epoch_end(self, epoch:int, num_batch:int, smooth_loss:Tensor,
                     last_metrics=MetricsList, **kwargs:Any)->bool:
        "Save epoch info: num_batch, smooth_loss, metrics."
        self.nb_batches.append(num_batch)
        if last_metrics is not None: self.val_losses.append(last_metrics[0])
        else: last_metrics = [] if self.no_val else [None]
        if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
        self.format_stats([epoch, smooth_loss] + last_metrics)

    def format_stats(self, stats:TensorOrNumList)->None:
        "Format stats before printing."
        str_stats = []
        for name,stat in zip(self.names,stats):
            str_stats.append('#na#' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.6f}')
        if self.add_time: str_stats.append(format_time(time() - self.start_epoch))
        if not self.silent: self.pbar.write(str_stats, table=True)

    def add_metric_names(self, names):
        "Add `names` to the inner metric names."
        if hasattr(self, '_added_met_names'): self._added_met_names += names
        else:                                 self._added_met_names  = names

    def plot_lr(self, show_moms=False, skip_start:int=0, skip_end:int=0, return_fig:bool=None)->Optional[plt.Figure]:
        "Plot learning rate, `show_moms` to include momentum."
        lrs = self._split_list(self.lrs, skip_start, skip_end)
        iterations = self._split_list(range_of(self.lrs), skip_start, skip_end)
        if show_moms:
            moms = self._split_list(self.moms, skip_start, skip_end)
            fig, axs = plt.subplots(1,2, figsize=(12,4))
            axs[0].plot(iterations, lrs)
            axs[0].set_xlabel('Iterations')
            axs[0].set_ylabel('Learning Rate')
            axs[1].plot(iterations, moms)
            axs[1].set_xlabel('Iterations')
            axs[1].set_ylabel('Momentum')
        else:
            fig, ax = plt.subplots()
            ax.plot(iterations, lrs)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Learning Rate')
        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)

    @staticmethod
    def smoothen_by_spline(xs, ys, **kwargs):
        xs = np.arange(len(ys))
        spl = scipy.interpolate.UnivariateSpline(xs, ys, **kwargs)
        ys = spl(xs)
        return ys

    def plot(self, skip_start:int=10, skip_end:int=5, suggestion:bool=False, return_fig:bool=None,
             **kwargs)->Optional[plt.Figure]:
        "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
        lrs = self._split_list(self.lrs, skip_start, skip_end)
        losses = self._split_list(self.losses, skip_start, skip_end)
        losses = [x.item() for x in losses]
        if 'k' in kwargs: losses = self.smoothen_by_spline(lrs, losses, **kwargs)
        fig, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        if suggestion:
            try: mg = (np.gradient(np.array(losses))).argmin()
            except:
                print("Failed to compute the gradients, there might not be enough points.")
                return
            print(f"Min numerical gradient: {lrs[mg]:.2E}")
            ax.plot(lrs[mg],losses[mg],markersize=10,marker='o',color='red')
            self.min_grad_lr = lrs[mg]
            ml = np.argmin(losses)
            print(f"Min loss divided by 10: {lrs[ml]/10:.2E}")
        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)

    def plot_losses(self, skip_start:int=0, skip_end:int=0, return_fig:bool=None)->Optional[plt.Figure]:
        "Plot training and validation losses."
        fig, ax = plt.subplots(1,1)
        losses = self._split_list(self.losses, skip_start, skip_end)
        iterations = self._split_list(range_of(self.losses), skip_start, skip_end)
        ax.plot(iterations, losses, label='Train')
        val_iter = self._split_list_val(np.cumsum(self.nb_batches), skip_start, skip_end)
        val_losses = self._split_list_val(self.val_losses, skip_start, skip_end)
        ax.plot(val_iter, val_losses, label='Validation')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Batches processed')
        ax.legend()
        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)

    def plot_metrics(self, skip_start:int=0, skip_end:int=0, return_fig:bool=None)->Optional[plt.Figure]:
        "Plot metrics collected during training."
        assert len(self.metrics) != 0, "There are no metrics to plot."
        fig, axes = plt.subplots(len(self.metrics[0]),1,figsize=(6, 4*len(self.metrics[0])))
        val_iter = self._split_list_val(np.cumsum(self.nb_batches), skip_start, skip_end)
        axes = axes.flatten() if len(self.metrics[0]) != 1 else [axes]
        for i, ax in enumerate(axes):
            values = [met[i] for met in self.metrics]
            values = self._split_list_val(values, skip_start, skip_end)
            ax.plot(val_iter, values)
            ax.set_ylabel(str(self.metrics_names[i]))
            ax.set_xlabel('Batches processed')             
        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)

    def _split_list(self, vals:Collection[float], skip_start:int, skip_end:int):
        return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]

    def _split_list_val(self, vals:Collection[float], skip_start:int, skip_end:int):
        val_iter = np.cumsum(self.nb_batches)
        start_val = (val_iter - skip_start >= 0).nonzero()[0].min()
        end_val = (val_iter[-1] - val_iter - skip_end >= 0).nonzero()[0].max()+1
        return vals[start_val:end_val] if skip_end > 0 else vals[start_val:]

class FakeOptimizer():
    def step(self): pass
    def zero_grad(self): pass

def load_callback(class_func, state, learn:Learner):
    init_kwargs, others = split_kwargs_by_func(state, class_func.__init__)
    res = class_func(learn, **init_kwargs) if issubclass(class_func, LearnerCallback) else class_func(**init_kwargs)
    for k,v in others.items(): setattr(res, k, v)
    return res

def load_learner(path:PathOrStr, file:PathLikeOrBinaryStream='export.pkl', test:ItemList=None, **db_kwargs):
    "Load a `Learner` object saved with `export_state` in `path/file` with empty data, optionally add `test` and load on `cpu`. `file` can be file-like (file or buffer)"
    source = Path(path)/file if is_pathlike(file) else file
    state = torch.load(source, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(source)
    model = state.pop('model')
    src = LabelLists.load_state(path, state.pop('data'))
    if test is not None: src.add_test(test)
    data = src.databunch(**db_kwargs)
    cb_state = state.pop('cb_state')
    clas_func = state.pop('cls')
    res = clas_func(data, model, **state)
    res.callback_fns = state['callback_fns'] #to avoid duplicates
    res.callbacks = [load_callback(c,s, res) for c,s in cb_state.items()]
    return res
