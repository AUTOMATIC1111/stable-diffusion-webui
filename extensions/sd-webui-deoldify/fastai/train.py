"Provides advanced training extensions to `fastai.basic_train`. Includes half-precision, learning rate finder, mixup, and one-cycle"
from .torch_core import *
from .callbacks import *
from .basic_data import *
from .basic_train import *

__all__ = ['BnFreeze', 'GradientClipping', 'ShowGraph', 'Interpretation', 'ClassificationInterpretation', 'MultiLabelClassificationInterpretation',
 'fit_one_cycle', 'lr_find', 'one_cycle_scheduler', 'to_fp16', 'to_fp32', 'mixup', 'AccumulateScheduler']

def one_cycle_scheduler(lr_max:float, **kwargs:Any)->OneCycleScheduler:
    "Instantiate a `OneCycleScheduler` with `lr_max`."
    return partial(OneCycleScheduler, lr_max=lr_max, **kwargs)

def fit_one_cycle(learn:Learner, cyc_len:int, max_lr:Union[Floats,slice]=defaults.lr,
                  moms:Tuple[float,float]=(0.95,0.85), div_factor:float=25., pct_start:float=0.3, final_div:float=None,
                  wd:float=None, callbacks:Optional[CallbackList]=None, tot_epochs:int=None, start_epoch:int=None,
                  batch_multiplier:int=1)->None:
    "Fit a model following the 1cycle policy."
    max_lr = learn.lr_range(max_lr)
    callbacks = listify(callbacks)
    callbacks.append(OneCycleScheduler(learn, max_lr, moms=moms, div_factor=div_factor, pct_start=pct_start,
                                       final_div=final_div, tot_epochs=tot_epochs, start_epoch=start_epoch))
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks, batch_multiplier=batch_multiplier)

def lr_find(learn:Learner, start_lr:Floats=1e-7, end_lr:Floats=10, num_it:int=100, stop_div:bool=True, wd:float=None,
        batch_multiplier:int=1):
    "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`. If `stop_div`, stops when loss diverges."
    start_lr = learn.lr_range(start_lr)
    start_lr = np.array(start_lr) if is_listy(start_lr) else start_lr
    end_lr = learn.lr_range(end_lr)
    end_lr = np.array(end_lr) if is_listy(end_lr) else end_lr
    cb = LRFinder(learn, start_lr, end_lr, num_it, stop_div)
    epochs = int(np.ceil(num_it/len(learn.data.train_dl)))
    learn.fit(epochs, start_lr, callbacks=[cb], wd=wd, batch_multiplier=batch_multiplier)

def to_fp16(learn:Learner, loss_scale:float=None, max_noskip:int=1000, dynamic:bool=True, clip:float=None,
            flat_master:bool=False, max_scale:float=2**24)->Learner:
    "Put `learn` in FP16 precision mode."
    learn.to_fp32()
    learn.model = model2half(learn.model)
    learn.data.add_tfm(batch_to_half)
    learn.mp_cb = MixedPrecision(learn, loss_scale=loss_scale, max_noskip=max_noskip, dynamic=dynamic, clip=clip,
                                 flat_master=flat_master, max_scale=max_scale)
    learn.callbacks.append(learn.mp_cb)
    return learn

def to_fp32(learn:Learner):
    "Put `learn` back to FP32 precision mode."
    learn.data.remove_tfm(batch_to_half)
    for cb in learn.callbacks:
        if isinstance(cb, MixedPrecision): learn.callbacks.remove(cb)
    learn.model = learn.model.float()
    return learn

def mixup(learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True) -> Learner:
    "Add mixup https://arxiv.org/abs/1710.09412 to `learn`."
    learn.callback_fns.append(partial(MixUpCallback, alpha=alpha, stack_x=stack_x, stack_y=stack_y))
    return learn

Learner.fit_one_cycle = fit_one_cycle
Learner.lr_find = lr_find
Learner.to_fp16 = to_fp16
Learner.to_fp32 = to_fp32
Learner.mixup = mixup

class ShowGraph(LearnerCallback):
    "Update a graph of learner stats and metrics after each epoch."
    def on_epoch_end(self, n_epochs:int, last_metrics:MetricsList, **kwargs)->bool:
        "If we have `last_metrics` plot them in our pbar graph"
        if last_metrics is not None and last_metrics[0] is not None:
            rec = self.learn.recorder
            iters = range_of(rec.losses)
            val_iter = np.array(rec.nb_batches).cumsum()
            x_bounds = (0, (n_epochs - len(rec.nb_batches)) * rec.nb_batches[-1] + len(rec.losses))
            y_bounds = (0, max((max(Tensor(rec.losses)), max(Tensor(rec.val_losses)))))
            rec.pbar.update_graph([(iters, rec.losses), (val_iter, rec.val_losses)], x_bounds, y_bounds)
        return {}

class BnFreeze(LearnerCallback):
    "Freeze moving average statistics in all non-trainable batchnorm layers."
    def on_epoch_begin(self, **kwargs:Any)->None:
        "Put bn layers in eval mode just after `model.train()`."
        set_bn_eval(self.learn.model)

class GradientClipping(LearnerCallback):
    "Gradient clipping during training."
    def __init__(self, learn:Learner, clip:float = 0.):
        super().__init__(learn)
        self.clip = clip

    def on_backward_end(self, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip: nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)

def clip_grad(learn:Learner, clip:float=0.1)->Learner:
    "Add gradient clipping of `clip` during training."
    learn.callback_fns.append(partial(GradientClipping, clip=clip))
    return learn
Learner.clip_grad = clip_grad

class AccumulateScheduler(LearnerCallback):
    "Does accumlated step every nth step by accumulating gradients"

    def __init__(self, learn:Learner, n_step:int = 1, drop_last:bool = False):
        super().__init__(learn)
        self.n_step,self.drop_last = n_step,drop_last

    def on_train_begin(self, **kwargs):
        "check if loss is reduction"
        if hasattr(self.loss_func, "reduction") and (self.loss_func.reduction != "sum"):
             warn("For better gradients consider 'reduction=sum'")

    def on_epoch_begin(self, **kwargs):
        "init samples and batches, change optimizer"
        self.acc_samples, self.acc_batches = 0., 0.

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "accumulate samples and batches"
        self.acc_samples += last_input.shape[0]
        self.acc_batches += 1

    def on_backward_end(self, **kwargs):
        "accumulated step and reset samples, True will result in no stepping"
        if (self.acc_batches % self.n_step) == 0:
            for p in (self.learn.model.parameters()):
                if p.requires_grad: p.grad.div_(self.acc_samples)
            self.acc_samples = 0
        else: return {'skip_step':True, 'skip_zero':True}

    def on_epoch_end(self, **kwargs):
        "step the rest of the accumulated grads if not perfectly divisible"
        for p in (self.learn.model.parameters()):
                if p.requires_grad: p.grad.div_(self.acc_samples)
        if not self.drop_last: self.learn.opt.step()
        self.learn.opt.zero_grad()


class Interpretation():
    "Interpretation base class, can be inherited for task specific Interpretation classes"
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid):
        self.data,self.preds,self.y_true,self.losses,self.ds_type, self.learn = \
                                 learn.data,preds,y_true,losses,ds_type,learn
        self.ds = (self.data.train_ds if ds_type == DatasetType.Train else
                   self.data.test_ds if ds_type == DatasetType.Test else
                   self.data.valid_ds if ds_type == DatasetType.Valid else
                   self.data.single_ds if ds_type == DatasetType.Single else
                   self.data.fix_ds)

    @classmethod
    def from_learner(cls, learn: Learner,  ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None):
        "Gets preds, y_true, losses to construct base class from a learner"
        preds_res = learn.get_preds(ds_type=ds_type, activ=activ, with_loss=True)
        return cls(learn, *preds_res)

    def top_losses(self, k:int=None, largest=True):
        "`k` largest(/smallest) losses and indexes, defaulting to all losses (sorted by `largest`)."
        return self.losses.topk(ifnone(k, len(self.losses)), largest=largest)

    # def top_scores(self, metric:Callable=None, k:int=None, largest=True):
    #     "`k` largest(/smallest) metric scores and indexes, defaulting to all scores (sorted by `largest`)."
    #     self.scores = metric(self.preds, self.y_true)
    #     return self.scores.topk(ifnone(k, len(self.scores)), largest=largest)


class ClassificationInterpretation(Interpretation):
    "Interpretation methods for classification models."
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid):
        super(ClassificationInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        self.pred_class = self.preds.argmax(dim=1)

    def confusion_matrix(self, slice_size:int=1):
        "Confusion matrix as an `np.ndarray`."
        x=torch.arange(0,self.data.c)
        if slice_size is None: cm = ((self.pred_class==x[:,None]) & (self.y_true==x[:,None,None])).sum(2)
        else:
            cm = torch.zeros(self.data.c, self.data.c, dtype=x.dtype)
            for i in range(0, self.y_true.shape[0], slice_size):
                cm_slice = ((self.pred_class[i:i+slice_size]==x[:,None])
                            & (self.y_true[i:i+slice_size]==x[:,None,None])).sum(2)
                torch.add(cm, cm_slice, out=cm)
        return to_np(cm)

    def plot_confusion_matrix(self, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", slice_size:int=1,
                              norm_dec:int=2, plot_txt:bool=True, return_fig:bool=None, **kwargs)->Optional[plt.Figure]:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix(slice_size=slice_size)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.data.c)
        plt.xticks(tick_marks, self.data.y.classes, rotation=90)
        plt.yticks(tick_marks, self.data.y.classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        if ifnone(return_fig, defaults.return_fig): return fig

    def most_confused(self, min_val:int=1, slice_size:int=1)->Collection[Tuple[str,str,int]]:
        "Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences."
        cm = self.confusion_matrix(slice_size=slice_size)
        np.fill_diagonal(cm, 0)
        res = [(self.data.classes[i],self.data.classes[j],cm[i,j])
                for i,j in zip(*np.where(cm>=min_val))]
        return sorted(res, key=itemgetter(2), reverse=True)


def _learner_interpret(learn:Learner, ds_type:DatasetType=DatasetType.Valid):
    "Create a `ClassificationInterpretation` object from `learner` on `ds_type` with `tta`."
    return ClassificationInterpretation.from_learner(learn, ds_type=ds_type)
Learner.interpret = _learner_interpret

class MultiLabelClassificationInterpretation(Interpretation):
    "Interpretation methods for classification models."
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid,
                     sigmoid:bool=True, thresh:float=0.3):
        raise NotImplementedError
        super(MultiLabelClassificationInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        self.pred_class = self.preds.sigmoid(dim=1)>thresh if sigmoid else self.preds>thresh
