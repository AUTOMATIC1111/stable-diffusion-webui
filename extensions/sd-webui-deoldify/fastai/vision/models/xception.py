from ...vision import *

__all__ = ['xception']

def sep_conv(ni,nf,pad=None,pool=False,act=True):
    layers =  [nn.ReLU()] if act else []
    layers += [
        nn.Conv2d(ni,ni,3,1,1,groups=ni,bias=False),
        nn.Conv2d(ni,nf,1,bias=False),
        nn.BatchNorm2d(nf)
    ]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def conv(ni,nf,ks=1,stride=1, pad=None, act=True):
    if pad is None: pad=ks//2
    layers = [
        nn.Conv2d(ni,nf,ks,stride,pad,bias=False),
        nn.BatchNorm2d(nf),
    ]
    if act: layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ConvSkip(Module):
    def __init__(self,ni,nf=None,act=True):
        self.nf,self.ni = nf,ni
        if self.nf is None: self.nf = ni
        self.conv = conv(ni,nf,stride=2, act=False)
        self.m = nn.Sequential(
            sep_conv(ni,ni,act=act),
            sep_conv(ni,nf,pool=True)
        )

    def forward(self,x): return self.conv(x) + self.m(x)

def middle_flow(nf):
    layers = [sep_conv(nf,nf) for i in range(3)]
    return SequentialEx(*layers, MergeLayer())

def xception(c, k=8, n_middle=8):
    "Preview version of Xception network. Not tested yet - use at own risk. No pretrained model yet."
    layers = [
        conv(3, k*4, 3, 2),
        conv(k*4, k*8, 3),
        ConvSkip(k*8, k*16, act=False),
        ConvSkip(k*16, k*32),
        ConvSkip(k*32, k*91),
    ]
    for i in range(n_middle): layers.append(middle_flow(k*91))
    layers += [
        ConvSkip(k*91,k*128),
        sep_conv(k*128,k*192,act=False),
        sep_conv(k*192,k*256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(k*256,c)
    ]
    return nn.Sequential(*layers)

