# TEED: is a Tiny but Efficient Edge Detection, it comes from the LDC-B3
# with a Slightly modification
# LDC parameters:
# 155665
# TED > 58K

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Fsmish import smish as Fsmish
from .Xsmish import Smish


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):
    # from LDC

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        self.conv3= nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32) # before 64

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class CoFusion2(nn.Module):
        # TEDv14-3
    def __init__(self, in_ch, out_ch):
        super(CoFusion2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3,
        #                        stride=1, padding=1)# before 64
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.smish= Smish()#nn.ReLU(inplace=True)


    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.conv1(self.smish(x))
        attn = self.conv3(self.smish(attn)) # before , )dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)

class DoubleFusion(nn.Module):
    # TED fusion before the final edge map prediction
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch*8, kernel_size=3,
                               stride=1, padding=1, groups=in_ch) # before 64
        self.PSconv1 = nn.PixelShuffle(1)

        self.DWconv2 = nn.Conv2d(24, 24*1, kernel_size=3,
                               stride=1, padding=1,groups=24)# before 64  instead of 32

        self.AF= Smish()#XAF() #nn.Tanh()# XAF() #   # Smish()#


    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.PSconv1(self.DWconv1(self.AF(x))) # #TEED best res TEDv14 [8, 32, 352, 352]

        attn2 = self.PSconv1(self.DWconv2(self.AF(attn))) # #TEED best res TEDv14[8, 3, 352, 352]

        return Fsmish(((attn2 +attn).sum(1)).unsqueeze(1)) #TED best res

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('smish1', Smish()),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))
    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(Fsmish(x1))  # F.relu()

        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        # self.use_bn = use_bs
        self.use_ac=use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish= Smish()#nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class TED(nn.Module):
    """ Definition of  Tiny and Efficient Edge Detector
    model
    """

    def __init__(self):
        super(TED, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48) # [32,48,100,100] before (2, 32, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # skip1 connection, see fig. 2
        self.side_1 = SingleConvBlock(16, 32, 2)

        # skip2 connection, see fig. 2
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)  # before (32, 64, 1)

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2) # (32, 64, 1)

        self.block_cat = DoubleFusion(3,3) # TEED: DoubleFusion

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        img_h, img_w = slice_shape
        if img_w!=t_shape[-1] or img_h!=t_shape[2]:
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic',align_corners=False)

        else:
            new_tensor=tensor
        # tensor[..., :height, :width]
        return new_tensor
    def resize_input(self,tensor):
        t_shape = tensor.shape
        if t_shape[2] % 8 != 0 or t_shape[3] % 8 != 0:
            img_w= ((t_shape[3]// 8) + 1) * 8
            img_h = ((t_shape[2] // 8) + 1) * 8
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def crop_bdcn(data1, h, w, crop_h, crop_w):
        # Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
        _, _, h1, w1 = data1.size()
        assert (h <= h1 and w <= w1)
        data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
        return data


    def forward(self, x, single_test=False):
        assert x.ndim == 4, x.shape
         # supose the image size is 352x352

        # Block 1
        block_1 = self.block_1(x) # [8,16,176,176]
        block_1_side = self.side_1(block_1) # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1) # 32 # [8,32,176,176]
        block_2_down = self.maxpool(block_2) # [8,32,88,88]
        block_2_add = block_2_down + block_1_side # [8,32,88,88]

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down) # [8,64,88,88] block 3 L connection
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense]) # [8,64,88,88]

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        results = [out_1, out_2, out_3]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW DoubleFusion

        results.append(block_cat)
        return results


if __name__ == '__main__':
    batch_size = 8
    img_height = 352
    img_width = 352

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = TED().to(device)
    output = model(input)
    print(f"output shapes: {[t.shape for t in output]}")

    # for i in range(20000):
    #     print(i)
    #     output = model(input)
    #     loss = nn.MSELoss()(output[-1], target)
    #     loss.backward()
