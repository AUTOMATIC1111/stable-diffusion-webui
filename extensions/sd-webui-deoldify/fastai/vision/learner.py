"`Learner` support for computer vision"
from ..torch_core import *
from ..basic_train import *
from ..basic_data import *
from .image import *
from . import models
from ..callback import *
from ..layers import *
from ..callbacks.hooks import *
from ..train import ClassificationInterpretation

__all__ = ['cnn_learner', 'create_cnn', 'create_cnn_model', 'create_body', 'create_head', 'unet_learner']
# By default split models between first and second layer
def _default_split(m:nn.Module): return (m[1],)
# Split a resnet style model
def _resnet_split(m:nn.Module): return (m[0][6],m[1])
# Split squeezenet model on maxpool layers
def _squeezenet_split(m:nn.Module): return (m[0][0][5], m[0][0][8], m[1])
def _densenet_split(m:nn.Module): return (m[0][0][7],m[1])
def _vgg_split(m:nn.Module): return (m[0][0][22],m[1])
def _alexnet_split(m:nn.Module): return (m[0][0][6],m[1])

_default_meta    = {'cut':None, 'split':_default_split}
_resnet_meta     = {'cut':-2, 'split':_resnet_split }
_squeezenet_meta = {'cut':-1, 'split': _squeezenet_split}
_densenet_meta   = {'cut':-1, 'split':_densenet_split}
_vgg_meta        = {'cut':-1, 'split':_vgg_split}
_alexnet_meta    = {'cut':-1, 'split':_alexnet_split}

model_meta = {
    models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},
    models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},
    models.resnet152:{**_resnet_meta},

    models.squeezenet1_0:{**_squeezenet_meta},
    models.squeezenet1_1:{**_squeezenet_meta},

    models.densenet121:{**_densenet_meta}, models.densenet169:{**_densenet_meta},
    models.densenet201:{**_densenet_meta}, models.densenet161:{**_densenet_meta},
    models.vgg16_bn:{**_vgg_meta}, models.vgg19_bn:{**_vgg_meta},
    models.alexnet:{**_alexnet_meta}}

def cnn_config(arch):
    "Get the metadata associated with `arch`."
    #torch.backends.cudnn.benchmark = True
    return model_meta.get(arch, _default_meta)

def has_pool_type(m):
    if is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False

def create_body(arch:Callable, pretrained:bool=True, cut:Optional[Union[int, Callable]]=None):
    "Cut off the body of a typically pretrained `model` at `cut` (int) or cut the model as specified by `cut(model)` (function)."
    model = arch(pretrained=pretrained)
    cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int):      return nn.Sequential(*list(model.children())[:cut])
    elif isinstance(cut, Callable): return cut(model)
    else:                           raise NamedError("cut must be either integer or a function")


def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                concat_pool:bool=True, bn_final:bool=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

def create_cnn_model(base_arch:Callable, nc:int, cut:Union[int,Callable]=None, pretrained:bool=True,
                     lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,
                     bn_final:bool=False, concat_pool:bool=True):
    "Create custom convnet architecture"
    body = create_body(base_arch, pretrained, cut)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, nc, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    else: head = custom_head
    return nn.Sequential(body, head)

def cnn_learner(data:DataBunch, base_arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,
                lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,
                split_on:Optional[SplitFuncOrIdxList]=None, bn_final:bool=False, init=nn.init.kaiming_normal_,
                concat_pool:bool=True, **kwargs:Any)->Learner:
    "Build convnet style learner."
    meta = cnn_config(base_arch)
    model = create_cnn_model(base_arch, data.c, cut, pretrained, lin_ftrs, ps=ps, custom_head=custom_head,
        bn_final=bn_final, concat_pool=concat_pool)
    learn = Learner(data, model, **kwargs)
    learn.split(split_on or meta['split'])
    if pretrained: learn.freeze()
    if init: apply_init(model[1], init)
    return learn

def create_cnn(data, base_arch, **kwargs):
    warn("`create_cnn` is deprecated and is now named `cnn_learner`.")
    return cnn_learner(data, base_arch, **kwargs)

def unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, cut:Union[int,Callable]=None, **learn_kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    model = to_device(models.unet.DynamicUnet(body, n_classes=data.c, img_size=size, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle), data.device)
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

@classmethod
def _cl_int_from_learner(cls, learn:Learner, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, tta=False):
    "Create an instance of `ClassificationInterpretation`. `tta` indicates if we want to use Test Time Augmentation."
    preds = learn.TTA(ds_type=ds_type, with_loss=True) if tta else learn.get_preds(ds_type=ds_type, activ=activ, with_loss=True)

    return cls(learn, *preds, ds_type=ds_type)

def _test_cnn(m):
    if not isinstance(m, nn.Sequential) or not len(m) == 2: return False
    return isinstance(m[1][0], (AdaptiveConcatPool2d, nn.AdaptiveAvgPool2d))

def _cl_int_gradcam(self, idx, heatmap_thresh:int=16, image:bool=True):
    m = self.learn.model.eval()
    im,cl = self.learn.data.dl(DatasetType.Valid).dataset[idx]
    cl = int(cl)
    xb,_ = self.data.one_item(im, detach=False, denorm=False) #put into a minibatch of batch size = 1
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cl)].backward() 
    acts  = hook_a.stored[0].cpu() #activation maps
    if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
        if image:
            xb_im = Image(xb[0])
            _,ax = plt.subplots()
            sz = list(xb_im.shape[-2:])
            xb_im.show(ax,title=f"pred. class: {self.pred_class[idx]}, actual class: {self.learn.data.classes[cl]}")
            ax.imshow(mult, alpha=0.4, extent=(0,*sz[::-1],0),
              interpolation='bilinear', cmap='magma')
        return mult

ClassificationInterpretation.GradCAM =_cl_int_gradcam

def _cl_int_plot_top_losses(self, k, largest=True, figsize=(12,12), heatmap:bool=False, heatmap_thresh:int=16,
                            return_fig:bool=None)->Optional[plt.Figure]:
    "Show images in `top_losses` along with their prediction, actual, loss, and probability of actual class."
    assert not heatmap or _test_cnn(self.learn.model), "`heatmap=True` requires a model like `cnn_learner` produces."
    if heatmap is None: heatmap = _test_cnn(self.learn.model)
    tl_val,tl_idx = self.top_losses(k, largest)
    classes = self.data.classes
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k/cols)
    fig,axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('prediction/actual/loss/probability', weight='bold', size=14)
    for i,idx in enumerate(tl_idx):
        im,cl = self.data.dl(self.ds_type).dataset[idx]
        cl = int(cl)
        im.show(ax=axes.flat[i], title=
            f'{classes[self.pred_class[idx]]}/{classes[cl]} / {self.losses[idx]:.2f} / {self.preds[idx][cl]:.2f}')
        if heatmap:
            mult = self.GradCAM(idx,heatmap_thresh,image=False)
            if mult is not None:
                sz = list(im.shape[-2:])
                axes.flat[i].imshow(mult, alpha=0.6, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap='magma')                
    if ifnone(return_fig, defaults.return_fig): return fig

def _cl_int_plot_multi_top_losses(self, samples:int=3, figsize:Tuple[int,int]=(8,8), save_misclassified:bool=False):
    "Show images in `top_losses` along with their prediction, actual, loss, and probability of predicted class in a multilabeled dataset."
    if samples >20:
        print("Max 20 samples")
        return
    losses, idxs = self.top_losses(self.data.c)
    l_dim = len(losses.size())
    if l_dim == 1: losses, idxs = self.top_losses()
    infolist, ordlosses_idxs, mismatches_idxs, mismatches, losses_mismatches, mismatchescontainer = [],[],[],[],[],[]
    truthlabels = np.asarray(self.y_true, dtype=int)
    classes_ids = [k for k in enumerate(self.data.classes)]
    predclass = np.asarray(self.pred_class)
    for i,pred in enumerate(predclass):
        where_truth = np.nonzero((truthlabels[i]>0))[0]
        mismatch = np.all(pred!=where_truth)
        if mismatch:
            mismatches_idxs.append(i)
            if l_dim > 1 : losses_mismatches.append((losses[i][pred], i))
            else: losses_mismatches.append((losses[i], i))
        if l_dim > 1: infotup = (i, pred, where_truth, losses[i][pred], np.round(self.preds[i], decimals=3)[pred], mismatch)
        else: infotup = (i, pred, where_truth, losses[i], np.round(self.preds[i], decimals=3)[pred], mismatch)
        infolist.append(infotup)
    ds = self.data.dl(self.ds_type).dataset
    mismatches = ds[mismatches_idxs]
    ordlosses = sorted(losses_mismatches, key = lambda x: x[0], reverse=True)
    for w in ordlosses: ordlosses_idxs.append(w[1])
    mismatches_ordered_byloss = ds[ordlosses_idxs]
    print(f'{str(len(mismatches))} misclassified samples over {str(len(self.data.valid_ds))} samples in the validation set.')
    samples = min(samples, len(mismatches))
    for ima in range(len(mismatches_ordered_byloss)):
        mismatchescontainer.append(mismatches_ordered_byloss[ima][0])
    for sampleN in range(samples):
        actualclasses = ''
        for clas in infoList[ordlosses_idxs[sampleN]][2]:
            actualclasses = f'{actualclasses} -- {str(classes_ids[clas][1])}'
        imag = mismatches_ordered_byloss[sampleN][0]
        imag = show_image(imag, figsize=figsize)
        imag.set_title(f"""Predicted: {classes_ids[infoList[ordlosses_idxs[sampleN]][1]][1]} \nActual: {actualclasses}\nLoss: {infoList[ordlosses_idxs[sampleN]][3]}\nProbability: {infoList[ordlosses_idxs[sampleN]][4]}""",
                        loc='left')
        plt.show()
        if save_misclassified: return mismatchescontainer

ClassificationInterpretation.from_learner          = _cl_int_from_learner
ClassificationInterpretation.plot_top_losses       = _cl_int_plot_top_losses
ClassificationInterpretation.plot_multi_top_losses = _cl_int_plot_multi_top_losses
 

def _learner_interpret(learn:Learner, ds_type:DatasetType=DatasetType.Valid, tta=False):
    "Create a `ClassificationInterpretation` object from `learner` on `ds_type` with `tta`."
    return ClassificationInterpretation.from_learner(learn, ds_type=ds_type, tta=tta)
Learner.interpret = _learner_interpret
