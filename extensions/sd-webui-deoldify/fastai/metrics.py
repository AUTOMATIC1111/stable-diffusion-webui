"Implements various metrics to measure training accuracy"
from .torch_core import *
from .callback import *
from .layers import *
from .basic_train import LearnerCallback

__all__ = ['error_rate', 'accuracy', 'accuracy_thresh', 'dice', 'exp_rmspe', 'fbeta','FBeta', 'mse', 'mean_squared_error',
            'mae', 'mean_absolute_error', 'rmse', 'root_mean_squared_error', 'msle', 'mean_squared_logarithmic_error',
            'explained_variance', 'r2_score', 'top_k_accuracy', 'KappaScore', 'ConfusionMatrix', 'MatthewsCorreff',
            'Precision', 'Recall', 'R2Score', 'ExplainedVariance', 'ExpRMSPE', 'RMSE', 'Perplexity', 'AUROC', 'auc_roc_score', 
            'roc_curve', 'MultiLabelFbeta', 'foreground_acc']

def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()

def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Computes accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean()

def top_k_accuracy(input:Tensor, targs:Tensor, k:int=5)->Rank0Tensor:
    "Computes the Top-k accuracy (target is in the top k predictions)."
    input = input.topk(k=k, dim=-1)[1]
    targs = targs.unsqueeze(dim=-1).expand_as(input)
    return (input == targs).max(dim=-1)[0].float().mean()

def foreground_acc(input, target, void_code):
    "Computes non-background accuracy, e.g. camvid for multiclass segmentation"
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def error_rate(input:Tensor, targs:Tensor)->Rank0Tensor:
    "1 - `accuracy`"
    return 1 - accuracy(input, targs)

def dice(input:Tensor, targs:Tensor, iou:bool=False, eps:float=1e-8)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return (intersect / (union-intersect+eps) if union > 0 else union.new([1.]).squeeze())

def psnr(input:Tensor, targs:Tensor)->Rank0Tensor:
    return 10 * (1. / mean_squared_error(input, targs)).log10()

def exp_rmspe(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred)/targ
    return torch.sqrt((pct_var**2).mean())

def mean_absolute_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean absolute error between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    return torch.abs(targ - pred).mean()

def mean_squared_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean squared error between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    return F.mse_loss(pred, targ)

def root_mean_squared_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Root mean squared error between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    return torch.sqrt(F.mse_loss(pred, targ))

def mean_squared_logarithmic_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean squared logarithmic error between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    return F.mse_loss(torch.log(1 + pred), torch.log(1 + targ))

def explained_variance(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Explained variance between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    var_pct = torch.var(targ - pred) / torch.var(targ)
    return 1 - var_pct

def r2_score(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "R2 score (coefficient of determination) between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - u / d

class RegMetrics(Callback):
    "Stores predictions and targets to perform calculations on epoch end."
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        assert last_output.numel() == last_target.numel(), "Expected same numbers of elements in pred & targ"
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu()))

class R2Score(RegMetrics):
    "Computes the R2 score (coefficient of determination)."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, r2_score(self.preds, self.targs))

class ExplainedVariance(RegMetrics):
    "Computes the explained variance."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, explained_variance(self.preds, self.targs))

class RMSE(RegMetrics):
    "Computes the root mean squared error."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, root_mean_squared_error(self.preds, self.targs))

class ExpRMSPE(RegMetrics):
    "Computes the exponential of the root mean square error."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, exp_rmspe(self.preds, self.targs))

# Aliases
mse = mean_squared_error
mae = mean_absolute_error
msle = mean_squared_logarithmic_error
rmse = root_mean_squared_error

class ConfusionMatrix(Callback):
    "Computes the confusion matrix."

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = last_output.argmax(-1).view(-1).cpu()
        targs = last_target.cpu()
        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm

@dataclass
class CMScores(ConfusionMatrix):
    "Base class for metrics which rely on the calculation of the precision and/or recall score."
    average:Optional[str]="binary"      # `binary`, `micro`, `macro`, `weigthed` or None
    pos_label:int=1                     # 0 or 1
    eps:float=1e-9

    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        if self.average is None: return rec
        else:
            if self.average == "micro": weights = self._weights(avg="weighted")
            else: weights = self._weights(avg=self.average)
            return (rec * weights).sum()

    def _precision(self):
        prec = torch.diag(self.cm) / self.cm.sum(dim=0)
        if self.average is None: return prec
        else:
            weights = self._weights(avg=self.average)
            return (prec * weights).sum()

    def _weights(self, avg:str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1: return Tensor([0,1])
            else: return Tensor([1,0])
        elif avg == "micro": return self.cm.sum(dim=0) / self.cm.sum()
        elif avg == "macro": return torch.ones((self.n_classes,)) / self.n_classes
        elif avg == "weighted": return self.cm.sum(dim=1) / self.cm.sum()


class Recall(CMScores):
    "Computes the Recall."
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._recall())

class Precision(CMScores):
    "Computes the Precision."
    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, self._precision())

@dataclass
class FBeta(CMScores):
    "Computes the F`beta` score."
    beta:float=2

    def on_train_begin(self, **kwargs):
        self.n_classes = 0
        self.beta2 = self.beta ** 2
        self.avg = self.average
        if self.average != "micro": self.average = None

    def on_epoch_end(self, last_metrics, **kwargs):
        prec = self._precision()
        rec = self._recall()
        metric = (1 + self.beta2) * prec * rec / (prec * self.beta2 + rec + self.eps)
        metric[metric != metric] = 0  # removing potential "nan"s
        if self.avg: metric = (self._weights(avg=self.avg) * metric).sum()
        return add_metrics(last_metrics, metric)

    def on_train_end(self, **kwargs): self.average = self.avg

@dataclass
class KappaScore(ConfusionMatrix):
    "Computes the rate of agreement (Cohens Kappa)."
    weights:Optional[str]=None      # None, `linear`, or `quadratic`

    def on_epoch_end(self, last_metrics, **kwargs):
        sum0 = self.cm.sum(dim=0)
        sum1 = self.cm.sum(dim=1)
        expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
        if self.weights is None:
            w = torch.ones((self.n_classes, self.n_classes))
            w[self.x, self.x] = 0
        elif self.weights == "linear" or self.weights == "quadratic":
            w = torch.zeros((self.n_classes, self.n_classes))
            w += torch.arange(self.n_classes, dtype=torch.float)
            w = torch.abs(w - torch.t(w)) if self.weights == "linear" else (w - torch.t(w)) ** 2
        else: raise ValueError('Unknown weights. Expected None, "linear", or "quadratic".')
        k = torch.sum(w * self.cm) / torch.sum(w * expected)
        return add_metrics(last_metrics, 1-k)

@dataclass
class MatthewsCorreff(ConfusionMatrix):
    "Computes the Matthews correlation coefficient."
    def on_epoch_end(self, last_metrics, **kwargs):
        t_sum = self.cm.sum(dim=1)
        p_sum = self.cm.sum(dim=0)
        n_correct = torch.trace(self.cm)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - torch.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - torch.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - torch.dot(t_sum, t_sum)
        return add_metrics(last_metrics, cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp))

class Perplexity(Callback):
    "Perplexity metric for language models."
    def on_epoch_begin(self, **kwargs): self.loss,self.len = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.loss += last_target.size(1) * CrossEntropyFlat()(last_output, last_target)
        self.len += last_target.size(1)

    def on_epoch_end(self, last_metrics, **kwargs): 
        return add_metrics(last_metrics, torch.exp(self.loss / self.len))

def auc_roc_score(input:Tensor, targ:Tensor):
    "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
    fpr, tpr = roc_curve(input, targ)
    d = fpr[1:] - fpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.).sum(-1)

def roc_curve(input:Tensor, targ:Tensor):
    "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
    targ = (targ == 1)
    desc_score_indices = torch.flip(input.argsort(-1), [-1])
    input = input[desc_score_indices]
    targ = targ[desc_score_indices]
    d = input[1:] - input[:-1]
    distinct_value_indices = torch.nonzero(d).transpose(0,1)[0]
    threshold_idxs = torch.cat((distinct_value_indices, LongTensor([len(targ) - 1]).to(targ.device)))
    tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
    fps = (1 + threshold_idxs - tps)
    if tps[0] != 0 or fps[0] != 0:
        fps = torch.cat((LongTensor([0]), fps))
        tps = torch.cat((LongTensor([0]), tps))
    fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]
    return fpr, tpr

@dataclass
class AUROC(Callback):
    "Computes the area under the curve (AUC) score based on the receiver operator characteristic (ROC) curve. Restricted to binary classification tasks."
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = LongTensor([]), Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        last_output = F.softmax(last_output, dim=1)[:,-1]
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu().long()))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, auc_roc_score(self.preds, self.targs))

class MultiLabelFbeta(LearnerCallback):
    "Computes the fbeta score for multilabel classification"
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    _order = -20 
    def __init__(self, learn, beta=2, eps=1e-15, thresh=0.3, sigmoid=True, average="micro"):
        super().__init__(learn)
        self.eps, self.thresh, self.sigmoid, self.average, self.beta2 = \
            eps, thresh, sigmoid, average, beta**2

    def on_train_begin(self, **kwargs):
        self.c = self.learn.data.c
        if self.average != "none": self.learn.recorder.add_metric_names([f'{self.average}_fbeta'])
        else: self.learn.recorder.add_metric_names([f"fbeta_{c}" for c in self.learn.data.classes])

    def on_epoch_begin(self, **kwargs):
        dvc = self.learn.data.device
        self.tp = torch.zeros(self.c).to(dvc)
        self.total_pred = torch.zeros(self.c).to(dvc)
        self.total_targ = torch.zeros(self.c).to(dvc)
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        pred, targ = (last_output.sigmoid() if self.sigmoid else last_output) > self.thresh, last_target.byte()
        m = pred*targ
        self.tp += m.sum(0).float()
        self.total_pred += pred.sum(0).float()
        self.total_targ += targ.sum(0).float()
    
    def fbeta_score(self, precision, recall):
        return (1 + self.beta2)*(precision*recall)/((self.beta2*precision + recall) + self.eps)

    def on_epoch_end(self, last_metrics, **kwargs):
        self.total_pred += self.eps
        self.total_targ += self.eps
        if self.average == "micro":
            precision, recall = self.tp.sum() / self.total_pred.sum(), self.tp.sum() / self.total_targ.sum()
            res = self.fbeta_score(precision, recall)
        elif self.average == "macro":
            res = self.fbeta_score((self.tp / self.total_pred), (self.tp / self.total_targ)).mean()
        elif self.average == "weighted":
            scores = self.fbeta_score((self.tp / self.total_pred), (self.tp / self.total_targ))
            res = (scores*self.total_targ).sum() / self.total_targ.sum()
        elif self.average == "none":
            res = listify(self.fbeta_score((self.tp / self.total_pred), (self.tp / self.total_targ)))
        else:
            raise Exception("Choose one of the average types: [micro, macro, weighted, none]")
        
        return add_metrics(last_metrics, res)
