from fastai.basic_train import Learner
from fastai.core import *
from fastai.layers import NormType, conv_layer
from fastai.torch_core import *
from fastai.vision import *
from fastai.vision.data import ImageDataBunch
from fastai.vision.gan import AdaptiveLoss, accuracy_thresh_expand

_conv_args = dict(leaky=0.2, norm_type=NormType.Spectral)


def _conv(ni: int, nf: int, ks: int = 3, stride: int = 1, **kwargs):
    return conv_layer(ni, nf, ks=ks, stride=stride, **_conv_args, **kwargs)


def custom_gan_critic(
    n_channels: int = 3, nf: int = 256, n_blocks: int = 3, p: int = 0.15
):
    "Critic to train a `GAN`."
    layers = [_conv(n_channels, nf, ks=4, stride=2), nn.Dropout2d(p / 2)]
    for i in range(n_blocks):
        layers += [
            _conv(nf, nf, ks=3, stride=1),
            nn.Dropout2d(p),
            _conv(nf, nf * 2, ks=4, stride=2, self_attention=(i == 0)),
        ]
        nf *= 2
    layers += [
        _conv(nf, nf, ks=3, stride=1),
        _conv(nf, 1, ks=4, bias=False, padding=0, use_activ=False),
        Flatten(),
    ]
    return nn.Sequential(*layers)


def colorize_crit_learner(
    data: ImageDataBunch,
    loss_critic=AdaptiveLoss(nn.BCEWithLogitsLoss()),
    nf: int = 256,
) -> Learner:
    return Learner(
        data,
        custom_gan_critic(nf=nf),
        metrics=accuracy_thresh_expand,
        loss_func=loss_critic,
        wd=1e-3,
    )
