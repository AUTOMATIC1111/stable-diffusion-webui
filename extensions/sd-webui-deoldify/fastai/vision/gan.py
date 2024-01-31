from ..torch_core import *
from ..layers import *
from ..callback import *
from ..basic_data import *
from ..basic_train import Learner, LearnerCallback
from .image import Image
from .data import ImageList

__all__ = ['basic_critic', 'basic_generator', 'GANModule', 'GANLoss', 'GANTrainer', 'FixedGANSwitcher', 'AdaptiveGANSwitcher',
           'GANLearner', 'NoisyItem', 'GANItemList', 'gan_critic', 'AdaptiveLoss', 'accuracy_thresh_expand',
           'GANDiscriminativeLR']

def AvgFlatten():
    "Takes the average of the input."
    return Lambda(lambda x: x.mean(0).view(1))

def basic_critic(in_size:int, n_channels:int, n_features:int=64, n_extra_layers:int=0, **conv_kwargs):
    "A basic critic for images `n_channels` x `in_size` x `in_size`."
    layers = [conv_layer(n_channels, n_features, 4, 2, 1, leaky=0.2, norm_type=None, **conv_kwargs)]#norm_type=None?
    cur_size, cur_ftrs = in_size//2, n_features
    layers.append(nn.Sequential(*[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=0.2, **conv_kwargs) for _ in range(n_extra_layers)]))
    while cur_size > 4:
        layers.append(conv_layer(cur_ftrs, cur_ftrs*2, 4, 2, 1, leaky=0.2, **conv_kwargs))
        cur_ftrs *= 2 ; cur_size //= 2
    layers += [conv2d(cur_ftrs, 1, 4, padding=0), AvgFlatten()]
    return nn.Sequential(*layers)

def basic_generator(in_size:int, n_channels:int, noise_sz:int=100, n_features:int=64, n_extra_layers=0, **conv_kwargs):
    "A basic generator from `noise_sz` to images `n_channels` x `in_size` x `in_size`."
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < in_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [conv_layer(noise_sz, cur_ftrs, 4, 1, transpose=True, **conv_kwargs)]
    cur_size = 4
    while cur_size < in_size // 2:
        layers.append(conv_layer(cur_ftrs, cur_ftrs//2, 4, 2, 1, transpose=True, **conv_kwargs))
        cur_ftrs //= 2; cur_size *= 2
    layers += [conv_layer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True, **conv_kwargs) for _ in range(n_extra_layers)]
    layers += [conv2d_trans(cur_ftrs, n_channels, 4, 2, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

class GANModule(Module):
    "Wrapper around a `generator` and a `critic` to create a GAN."
    def __init__(self, generator:nn.Module=None, critic:nn.Module=None, gen_mode:bool=False):
        self.gen_mode = gen_mode
        if generator: self.generator,self.critic = generator,critic

    def forward(self, *args):
        return self.generator(*args) if self.gen_mode else self.critic(*args)

    def switch(self, gen_mode:bool=None):
        "Put the model in generator mode if `gen_mode`, in critic mode otherwise."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode

class GANLoss(GANModule):
    "Wrapper around `loss_funcC` (for the critic) and `loss_funcG` (for the generator)."
    def __init__(self, loss_funcG:Callable, loss_funcC:Callable, gan_model:GANModule):
        super().__init__()
        self.loss_funcG,self.loss_funcC,self.gan_model = loss_funcG,loss_funcC,gan_model

    def generator(self, output, target):
        "Evaluate the `output` with the critic then uses `self.loss_funcG` to combine it with `target`."
        fake_pred = self.gan_model.critic(output)
        return self.loss_funcG(fake_pred, target, output)

    def critic(self, real_pred, input):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.loss_funcD`."
        fake = self.gan_model.generator(input.requires_grad_(False)).requires_grad_(True)
        fake_pred = self.gan_model.critic(fake)
        return self.loss_funcC(real_pred, fake_pred)

class GANTrainer(LearnerCallback):
    "Handles GAN Training."
    _order=-20
    def __init__(self, learn:Learner, switch_eval:bool=False, clip:float=None, beta:float=0.98, gen_first:bool=False,
                 show_img:bool=True):
        super().__init__(learn)
        self.switch_eval,self.clip,self.beta,self.gen_first,self.show_img = switch_eval,clip,beta,gen_first,show_img
        self.generator,self.critic = self.model.generator,self.model.critic

    def _set_trainable(self):
        train_model = self.generator if     self.gen_mode else self.critic
        loss_model  = self.generator if not self.gen_mode else self.critic
        requires_grad(train_model, True)
        requires_grad(loss_model, False)
        if self.switch_eval:
            train_model.train()
            loss_model.eval()

    def on_train_begin(self, **kwargs):
        "Create the optimizers for the generator and critic if necessary, initialize smootheners."
        if not getattr(self,'opt_gen',None):
            self.opt_gen = self.opt.new([nn.Sequential(*flatten_model(self.generator))])
        else: self.opt_gen.lr,self.opt_gen.wd = self.opt.lr,self.opt.wd
        if not getattr(self,'opt_critic',None):
            self.opt_critic = self.opt.new([nn.Sequential(*flatten_model(self.critic))])
        else: self.opt_critic.lr,self.opt_critic.wd = self.opt.lr,self.opt.wd
        self.gen_mode = self.gen_first
        self.switch(self.gen_mode)
        self.closses,self.glosses = [],[]
        self.smoothenerG,self.smoothenerC = SmoothenValue(self.beta),SmoothenValue(self.beta)
        #self.recorder.no_val=True
        self.recorder.add_metric_names(['gen_loss', 'disc_loss'])
        self.imgs,self.titles = [],[]

    def on_train_end(self, **kwargs):
        "Switch in generator mode for showing results."
        self.switch(gen_mode=True)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Clamp the weights with `self.clip` if it's not None, return the correct input."
        if self.clip is not None:
            for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
        return {'last_input':last_input,'last_target':last_target} if self.gen_mode else {'last_input':last_target,'last_target':last_input}

    def on_backward_begin(self, last_loss, last_output, **kwargs):
        "Record `last_loss` in the proper list."
        last_loss = last_loss.detach().cpu()
        if self.gen_mode:
            self.smoothenerG.add_value(last_loss)
            self.glosses.append(self.smoothenerG.smooth)
            self.last_gen = last_output.detach().cpu()
        else:
            self.smoothenerC.add_value(last_loss)
            self.closses.append(self.smoothenerC.smooth)

    def on_epoch_begin(self, epoch, **kwargs):
        "Put the critic or the generator back to eval if necessary."
        self.switch(self.gen_mode)

    def on_epoch_end(self, pbar, epoch, last_metrics, **kwargs):
        "Put the various losses in the recorder and show a sample image."
        if not hasattr(self, 'last_gen') or not self.show_img: return
        data = self.learn.data
        img = self.last_gen[0]
        norm = getattr(data,'norm',False)
        if norm and norm.keywords.get('do_y',False): img = data.denorm(img)
        img = data.train_ds.y.reconstruct(img)
        self.imgs.append(img)
        self.titles.append(f'Epoch {epoch}')
        pbar.show_imgs(self.imgs, self.titles)
        return add_metrics(last_metrics, [getattr(self.smoothenerG,'smooth',None),getattr(self.smoothenerC,'smooth',None)])

    def switch(self, gen_mode:bool=None):
        "Switch the model, if `gen_mode` is provided, in the desired mode."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
        self.opt.opt = self.opt_gen.opt if self.gen_mode else self.opt_critic.opt
        self._set_trainable()
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)

class FixedGANSwitcher(LearnerCallback):
    "Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator."
    def __init__(self, learn:Learner, n_crit:Union[int,Callable]=1, n_gen:Union[int,Callable]=1):
        super().__init__(learn)
        self.n_crit,self.n_gen = n_crit,n_gen

    def on_train_begin(self, **kwargs):
        "Initiate the iteration counts."
        self.n_c,self.n_g = 0,0

    def on_batch_end(self, iteration, **kwargs):
        "Switch the model if necessary."
        if self.learn.gan_trainer.gen_mode:
            self.n_g += 1
            n_iter,n_in,n_out = self.n_gen,self.n_c,self.n_g
        else:
            self.n_c += 1
            n_iter,n_in,n_out = self.n_crit,self.n_g,self.n_c
        target = n_iter if isinstance(n_iter, int) else n_iter(n_in)
        if target == n_out:
            self.learn.gan_trainer.switch()
            self.n_c,self.n_g = 0,0

@dataclass
class AdaptiveGANSwitcher(LearnerCallback):
    "Switcher that goes back to generator/critic when the loss goes below `gen_thresh`/`crit_thresh`."
    def __init__(self, learn:Learner, gen_thresh:float=None, critic_thresh:float=None):
        super().__init__(learn)
        self.gen_thresh,self.critic_thresh = gen_thresh,critic_thresh

    def on_batch_end(self, last_loss, **kwargs):
        "Switch the model if necessary."
        if self.gan_trainer.gen_mode:
            if self.gen_thresh  is None:      self.gan_trainer.switch()
            elif last_loss < self.gen_thresh: self.gan_trainer.switch()
        else:
            if self.critic_thresh is None:       self.gan_trainer.switch()
            elif last_loss < self.critic_thresh: self.gan_trainer.switch()

def gan_loss_from_func(loss_gen, loss_crit, weights_gen:Tuple[float,float]=None):
    "Define loss functions for a GAN from `loss_gen` and `loss_crit`."
    def _loss_G(fake_pred, output, target, weights_gen=weights_gen):
        ones = fake_pred.new_ones(fake_pred.shape[0])
        weights_gen = ifnone(weights_gen, (1.,1.))
        return weights_gen[0] * loss_crit(fake_pred, ones) + weights_gen[1] * loss_gen(output, target)

    def _loss_C(real_pred, fake_pred):
        ones  = real_pred.new_ones (real_pred.shape[0])
        zeros = fake_pred.new_zeros(fake_pred.shape[0])
        return (loss_crit(real_pred, ones) + loss_crit(fake_pred, zeros)) / 2

    return _loss_G, _loss_C

class GANLearner(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self, data:DataBunch, generator:nn.Module, critic:nn.Module, gen_loss_func:LossFunction,
                 crit_loss_func:LossFunction, switcher:Callback=None, gen_first:bool=False, switch_eval:bool=True,
                 show_img:bool=True, clip:float=None, **learn_kwargs):
        gan = GANModule(generator, critic)
        loss_func = GANLoss(gen_loss_func, crit_loss_func, gan)
        switcher = ifnone(switcher, partial(FixedGANSwitcher, n_crit=5, n_gen=1))
        super().__init__(data, gan, loss_func=loss_func, callback_fns=[switcher], **learn_kwargs)
        trainer = GANTrainer(self, clip=clip, switch_eval=switch_eval, show_img=show_img)
        self.gan_trainer = trainer
        self.callbacks.append(trainer)

    @classmethod
    def from_learners(cls, learn_gen:Learner, learn_crit:Learner, switcher:Callback=None,
                      weights_gen:Tuple[float,float]=None, **learn_kwargs):
        "Create a GAN from `learn_gen` and `learn_crit`."
        losses = gan_loss_from_func(learn_gen.loss_func, learn_crit.loss_func, weights_gen=weights_gen)
        return cls(learn_gen.data, learn_gen.model, learn_crit.model, *losses, switcher=switcher, **learn_kwargs)

    @classmethod
    def wgan(cls, data:DataBunch, generator:nn.Module, critic:nn.Module, switcher:Callback=None, clip:float=0.01, **learn_kwargs):
        "Create a WGAN from `data`, `generator` and `critic`."
        return cls(data, generator, critic, NoopLoss(), WassersteinLoss(), switcher=switcher, clip=clip, **learn_kwargs)

class NoisyItem(ItemBase):
    "An random `ItemBase` of size `noise_sz`."
    def __init__(self, noise_sz): self.obj,self.data = noise_sz,torch.randn(noise_sz, 1, 1)
    def __str__(self):  return ''
    def apply_tfms(self, tfms, **kwargs): return self

class GANItemList(ImageList):
    "`ItemList` suitable for GANs."
    _label_cls = ImageList

    def __init__(self, items, noise_sz:int=100, **kwargs):
        super().__init__(items, **kwargs)
        self.noise_sz = noise_sz
        self.copy_new.append('noise_sz')

    def get(self, i): return NoisyItem(self.noise_sz)
    def reconstruct(self, t): return NoisyItem(t.size(0))

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `ys` (target images) on a figure of `figsize`."
        super().show_xys(ys, xs, imgsize=imgsize, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `zs` (generated images) on a figure of `figsize`."
        super().show_xys(zs, xs, imgsize=imgsize, figsize=figsize, **kwargs)

_conv_args = dict(leaky=0.2, norm_type=NormType.Spectral)

def _conv(ni:int, nf:int, ks:int=3, stride:int=1, **kwargs):
    return conv_layer(ni, nf, ks=ks, stride=stride, **_conv_args, **kwargs)

def gan_critic(n_channels:int=3, nf:int=128, n_blocks:int=3, p:int=0.15):
    "Critic to train a `GAN`."
    layers = [
        _conv(n_channels, nf, ks=4, stride=2),
        nn.Dropout2d(p/2),
        res_block(nf, dense=True,**_conv_args)]
    nf *= 2 # after dense block
    for i in range(n_blocks):
        layers += [
            nn.Dropout2d(p),
            _conv(nf, nf*2, ks=4, stride=2, self_attention=(i==0))]
        nf *= 2
    layers += [
        _conv(nf, 1, ks=4, bias=False, padding=0, use_activ=False),
        Flatten()]
    return nn.Sequential(*layers)

class GANDiscriminativeLR(LearnerCallback):
    "`Callback` that handles multiplying the learning rate by `mult_lr` for the critic."
    def __init__(self, learn:Learner, mult_lr:float = 5.):
        super().__init__(learn)
        self.mult_lr = mult_lr

    def on_batch_begin(self, train, **kwargs):
        "Multiply the current lr if necessary."
        if not self.learn.gan_trainer.gen_mode and train: self.learn.opt.lr *= self.mult_lr

    def on_step_end(self, **kwargs):
        "Put the LR back to its value if necessary."
        if not self.learn.gan_trainer.gen_mode: self.learn.opt.lr /= self.mult_lr

class AdaptiveLoss(Module):
    "Expand the `target` to match the `output` size before applying `crit`."
    def __init__(self, crit):
        self.crit = crit

    def forward(self, output, target):
        return self.crit(output, target[:,None].expand_as(output).float())

def accuracy_thresh_expand(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy after expanding `y_true` to the size of `y_pred`."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true[:,None].expand_as(y_pred).byte()).float().mean()
