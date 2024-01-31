#These models are dowloaded via the repo https://github.com/Cadene/pretrained-models.pytorch
#See licence here: https://github.com/Cadene/pretrained-models.pytorch/blob/master/LICENSE.txt
from torch import nn
from ..learner import model_meta
from ...core import *

pretrainedmodels = try_import('pretrainedmodels')
if not pretrainedmodels:
    raise Exception('Error: `pretrainedmodels` is needed. `pip install pretrainedmodels`')

__all__ = ['inceptionv4', 'inceptionresnetv2', 'nasnetamobile', 'dpn92', 'xception_cadene', 'se_resnet50',
           'se_resnet101', 'se_resnext50_32x4d', 'senet154', 'pnasnet5large', 'se_resnext101_32x4d']

def get_model(model_name:str, pretrained:bool, seq:bool=False, pname:str='imagenet', **kwargs):
    pretrained = pname if pretrained else None
    model = getattr(pretrainedmodels, model_name)(pretrained=pretrained, **kwargs)
    return nn.Sequential(*model.children()) if seq else model

def inceptionv4(pretrained:bool=False):
    model = get_model('inceptionv4', pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers[0], *all_layers[1:])
model_meta[inceptionv4] = {'cut': -2, 'split': lambda m: (m[0][11], m[1])}

def nasnetamobile(pretrained:bool=False):
    model = get_model('nasnetamobile', pretrained, num_classes=1000)
    model.logits = noop
    return nn.Sequential(model)
model_meta[nasnetamobile] = {'cut': noop, 'split': lambda m: (list(m[0][0].children())[8], m[1])}

def pnasnet5large(pretrained:bool=False):
    model = get_model('pnasnet5large', pretrained, num_classes=1000)
    model.logits = noop
    return nn.Sequential(model)
model_meta[pnasnet5large] = {'cut': noop, 'split': lambda m: (list(m[0][0].children())[8], m[1])}

def inceptionresnetv2(pretrained:bool=False):   return get_model('inceptionresnetv2', pretrained, seq=True)
def dpn92(pretrained:bool=False):               return get_model('dpn92', pretrained, pname='imagenet+5k', seq=True)
def xception_cadene(pretrained=False):          return get_model('xception', pretrained, seq=True)
def se_resnet50(pretrained:bool=False):         return get_model('se_resnet50', pretrained)
def se_resnet101(pretrained:bool=False):        return get_model('se_resnet101', pretrained)
def se_resnext50_32x4d(pretrained:bool=False):  return get_model('se_resnext50_32x4d', pretrained)
def se_resnext101_32x4d(pretrained:bool=False): return get_model('se_resnext101_32x4d', pretrained)
def senet154(pretrained:bool=False):            return get_model('senet154', pretrained)

model_meta[inceptionresnetv2] = {'cut': -2, 'split': lambda m: (m[0][9],     m[1])}
model_meta[dpn92]             = {'cut': -1, 'split': lambda m: (m[0][0][16], m[1])}
model_meta[xception_cadene]   = {'cut': -1, 'split': lambda m: (m[0][11],    m[1])}
model_meta[senet154]          = {'cut': -3, 'split': lambda m: (m[0][3],     m[1])}
_se_resnet_meta               = {'cut': -2, 'split': lambda m: (m[0][3],     m[1])}
model_meta[se_resnet50]         = _se_resnet_meta
model_meta[se_resnet101]        = _se_resnet_meta
model_meta[se_resnext50_32x4d]  = _se_resnet_meta
model_meta[se_resnext101_32x4d] = _se_resnet_meta

# TODO: add "resnext101_32x4d" "resnext101_64x4d" after serialization issue is fixed:
# https://github.com/Cadene/pretrained-models.pytorch/pull/128
