import os
import torch
import torch.nn as nn
from PIL import Image
import fnmatch
import cv2

import sys

import numpy as np
from einops import rearrange
from modules import devices
from annotator.annotator_path import models_path


class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)

        # the following are for debugs
        print("****", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
        for i,layer in enumerate(self.model):
            if i != 2:
                x = layer(x)
            else:
                x = layer(x)
                #x = nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
            print("____", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
            print(x[0])
        return x

class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)



class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(
                    nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        #print(x.size(), y.size(), self.process)
        if self.process:
            y0 = self.model(x)
            #print("merge+", torch.max(y0+y), torch.min(y0+y),torch.mean(y0+y), torch.std(y0+y), y0.shape)
            return y0 + y
        else:
            #print("merge", torch.max(x+y), torch.min(x+y),torch.mean(x+y), torch.std(x+y), y.shape)
            return x + y

class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)

class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            l = None
            if i == 0: 
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(nn.Module):

    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y


class MangaLineExtration:
    model_dir = os.path.join(models_path, "manga_line")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth"
        modelpath = os.path.join(self.model_dir, "erika.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        #norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = res_skip()
        ckpt = torch.load(modelpath)
        for key in list(ckpt.keys()):
            if 'module.' in key:
                ckpt[key.replace('module.', '')] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net.to(self.device)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)
        img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        img = np.ascontiguousarray(img.copy()).copy()
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(self.device)
            image_feed = rearrange(image_feed, 'h w -> 1 1 h w')
            line = self.model(image_feed)
            line = 255 - line.cpu().numpy()[0, 0]
            return line.clip(0, 255).astype(np.uint8)

    
