from fastai.layers import *
from .layers import *
from fastai.torch_core import *
from fastai.callbacks.hooks import *
from fastai.vision import *


# The code below is meant to be merged into fastaiv1 ideally

__all__ = ['DynamicUnetDeep', 'DynamicUnetWide']


def _get_sfs_idxs(sizes: Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(
        np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0]
    )
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float = None,
        **kwargs
    ):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(
            ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        hook: Hook,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        nf_factor: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = custom_conv_layer(
            nf, nf, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class DynamicUnetDeep(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final=True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        norm_type: Optional[NormType] = NormType.Batch,
        nf_factor: float = 1.0,
        **kwargs
    ):
        extra_bn = norm_type == NormType.Spectral
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            ),
        ).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block = UnetBlockDeep(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                nf_factor=nf_factor,
                **kwargs
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()


# ------------------------------------------------------
class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        n_out: int,
        hook: Hook,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hook = hook
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, x_out, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


class DynamicUnetWide(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final=True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        norm_type: Optional[NormType] = NormType.Batch,
        nf_factor: int = 1,
        **kwargs
    ):

        nf = 512 * nf_factor
        extra_bn = norm_type == NormType.Spectral
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs
            ),
        ).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)

            n_out = nf if not_final else nf // 2

            unet_block = UnetBlockWide(
                up_in_c,
                x_in_c,
                n_out,
                self.sfs[i],
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()
