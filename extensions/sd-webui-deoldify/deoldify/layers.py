from fastai.layers import *
from fastai.torch_core import *


# The code below is meant to be merged into fastaiv1 ideally


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[NormType] = NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)
