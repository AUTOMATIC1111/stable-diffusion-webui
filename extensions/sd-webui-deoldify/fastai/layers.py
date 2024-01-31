"`fastai.layers` provides essential functions to building and modifying `model` architectures"
from .torch_core import *

__all__ = ['AdaptiveConcatPool2d', 'BCEWithLogitsFlat', 'BCEFlat', 'MSELossFlat', 'CrossEntropyFlat', 'Debugger',
           'Flatten', 'Lambda', 'PoolFlatten', 'View', 'ResizeBatch', 'bn_drop_lin', 'conv2d', 'conv2d_trans', 'conv_layer',
           'embedding', 'simple_cnn', 'NormType', 'relu', 'batchnorm_2d', 'trunc_normal_', 'PixelShuffle_ICNR', 'icnr',
           'NoopLoss', 'WassersteinLoss', 'SelfAttention', 'SequentialEx', 'MergeLayer', 'res_block', 'sigmoid_range',
           'SigmoidRange', 'PartialLayer', 'FlattenedLoss', 'BatchNorm1dFlat', 'LabelSmoothingCrossEntropy', 'PooledSelfAttention2d']

class Lambda(Module):
    "Create a layer that simply calls `func` with `x`"
    def __init__(self, func:LambdaFunc): self.func=func
    def forward(self, x): return self.func(x)

class View(Module):
    "Reshape `x` to `size`"
    def __init__(self, *size:int): self.size = size
    def forward(self, x): return x.view(self.size)

class ResizeBatch(Module):
    "Reshape `x` to `size`, keeping batch dim the same size"
    def __init__(self, *size:int): self.size = size
    def forward(self, x): return x.view((x.size(0),) + self.size)

class Flatten(Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False): self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)

def PoolFlatten()->nn.Sequential:
    "Apply `nn.AdaptiveAvgPool2d` to `x` and then flatten the result."
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten())

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Group Instance SpectralGN')

def batchnorm_2d(nf:int, norm_type:NormType=NormType.Batch):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type==NormType.BatchZero else 1.)
    return bn

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

class PooledSelfAttention2d(Module):
    "Pooled self attention layer for 2d."
    def __init__(self, n_channels:int):
        self.n_channels = n_channels
        self.theta = spectral_norm(conv2d(n_channels, n_channels//8, 1)) # query
        self.phi   = spectral_norm(conv2d(n_channels, n_channels//8, 1)) # key
        self.g     = spectral_norm(conv2d(n_channels, n_channels//2, 1)) # value
        self.o     = spectral_norm(conv2d(n_channels//2, n_channels, 1))
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        # code borrowed from https://github.com/ajbrock/BigGAN-PyTorch/blob/7b65e82d058bfe035fc4e299f322a1f83993e04c/layers.py#L156
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        theta = theta.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.n_channels // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.n_channels // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class SelfAttention(Module):
    "Self attention layer for nd."
    def __init__(self, n_channels:int):
        self.query = conv1d(n_channels, n_channels//8)
        self.key   = conv1d(n_channels, n_channels//8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        #Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

def conv2d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False, init:LayerFunc=nn.init.kaiming_normal_) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None: padding = ks//2
    return init_default(nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)

def conv2d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0, bias=False) -> nn.ConvTranspose2d:
    "Create `nn.ConvTranspose2d` layer."
    return nn.ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def relu(inplace:bool=False, leaky:float=None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)

def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, is_1d:bool=False,
               norm_type:Optional[NormType]=NormType.Batch,  use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)

class SequentialEx(Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"
    def __init__(self, *layers): self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            #print(l. + ' mean: ' + str(nres.abs().mean()))
            #print(' max: ' + str(nres.abs().max()))
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self,i): return self.layers[i]
    def append(self,l): return self.layers.append(l)
    def extend(self,l): return self.layers.extend(l)
    def insert(self,i,l): return self.layers.insert(i,l)

class MergeLayer(Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."
    def __init__(self, dense:bool=False): self.dense=dense
    def forward(self, x): return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)

def res_block(nf, dense:bool=False, norm_type:Optional[NormType]=NormType.Batch, bottle:bool=False, **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf//2 if bottle else nf
    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                      conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                      MergeLayer(dense))

def sigmoid_range(x:Tensor, low:int, high:int):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

class SigmoidRange(Module):
    "Sigmoid module with range `(low,x_max)`"
    def __init__(self, low:int, high:int): self.low,self.high = low,high
    def forward(self, x): return sigmoid_range(x, self.low, self.high)

class PartialLayer(Module):
    "Layer that applies `partial(func, **kwargs)`."
    def __init__(self, func, **kwargs): self.repr,self.func = f'{func}({kwargs})', partial(func, **kwargs)
    def forward(self, x): return self.func(x)
    def __repr__(self): return self.repr

class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Debugger(Module):
    "A module to debug inside a model."
    def forward(self,x:Tensor) -> Tensor:
        set_trace()
        return x

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle_ICNR(Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=False, norm_type=NormType.Weight, leaky:float=None):
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni, nf*(scale**2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x

class FlattenedLoss():
    "Same as `func`, but flattens input and target."
    def __init__(self, func, *args, axis:int=-1, floatify:bool=False, is_2d:bool=True, **kwargs):
        self.func,self.axis,self.floatify,self.is_2d = func(*args,**kwargs),axis,floatify,is_2d
        functools.update_wrapper(self, self.func)

    def __repr__(self): return f"FlattenedLoss of {self.func}"
    @property
    def reduction(self): return self.func.reduction
    @reduction.setter
    def reduction(self, v): self.func.reduction = v

    def __call__(self, input:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        input = input.transpose(self.axis,-1).contiguous()
        target = target.transpose(self.axis,-1).contiguous()
        if self.floatify: target = target.float()
        input = input.view(-1,input.shape[-1]) if self.is_2d else input.view(-1)
        return self.func.__call__(input, target.view(-1), **kwargs)

def CrossEntropyFlat(*args, axis:int=-1, **kwargs):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    return FlattenedLoss(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)

def BCEWithLogitsFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.BCEWithLogitsLoss`, but flattens input and target."
    return FlattenedLoss(nn.BCEWithLogitsLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

def BCEFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.BCELoss`, but flattens input and target."
    return FlattenedLoss(nn.BCELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

def MSELossFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.MSELoss`, but flattens input and target."
    return FlattenedLoss(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

class NoopLoss(Module):
    "Just returns the mean of the `output`."
    def forward(self, output, *args): return output.mean()

class WassersteinLoss(Module):
    "For WGAN."
    def forward(self, real, fake): return real.mean() - fake.mean()

def simple_cnn(actns:Collection[int], kernel_szs:Collection[int]=None,
               strides:Collection[int]=None, bn=False) -> nn.Sequential:
    "CNN with `conv_layer` defined by `actns`, `kernel_szs` and `strides`, plus batchnorm if `bn`."
    nl = len(actns)-1
    kernel_szs = ifnone(kernel_szs, [3]*nl)
    strides    = ifnone(strides   , [2]*nl)
    layers = [conv_layer(actns[i], actns[i+1], kernel_szs[i], stride=strides[i],
              norm_type=(NormType.Batch if bn and i<(len(strides)-1) else None)) for i in range_of(strides)]
    layers.append(PoolFlatten())
    return nn.Sequential(*layers)

def trunc_normal_(x:Tensor, mean:float=0., std:float=1.) -> Tensor:
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def embedding(ni:int,nf:int) -> nn.Module:
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb

class BatchNorm1dFlat(nn.BatchNorm1d):
    "`nn.BatchNorm1d`, but first flattens leading dimensions"
    def forward(self, x):
        if x.dim()==2: return super().forward(x)
        *f,l = x.shape
        x = x.contiguous().view(-1,l)
        return super().forward(x).view(*f,l)

class LabelSmoothingCrossEntropy(Module):
    def __init__(self, eps:float=0.1, reduction='mean'): self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)
