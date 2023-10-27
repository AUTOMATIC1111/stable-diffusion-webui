# Modified from Diffusers to reduce VRAM usage

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.autoencoder_kl import AutoencoderKLOutput


def slice_h(x, num_slices):
    # slice with pad 1 both sides: to eliminate side effect of padding of conv2d
    # Conv2dのpaddingの副作用を排除するために、両側にpad 1しながらHをスライスする
    # NCHWでもNHWCでもどちらでも動く
    size = (x.shape[2] + num_slices - 1) // num_slices
    sliced = []
    for i in range(num_slices):
        if i == 0:
            sliced.append(x[:, :, : size + 1, :])
        else:
            end = size * (i + 1) + 1
            if x.shape[2] - end < 3:  # if the last slice is too small, use the rest of the tensor 最後が細すぎるとconv2dできないので全部使う
                end = x.shape[2]
            sliced.append(x[:, :, size * i - 1 : end, :])
            if end >= x.shape[2]:
                break
    return sliced


def cat_h(sliced):
    # padding分を除いて結合する
    cat = []
    for i, x in enumerate(sliced):
        if i == 0:
            cat.append(x[:, :, :-1, :])
        elif i == len(sliced) - 1:
            cat.append(x[:, :, 1:, :])
        else:
            cat.append(x[:, :, 1:-1, :])
        del x
    x = torch.cat(cat, dim=2)
    return x


def resblock_forward(_self, num_slices, input_tensor, temb):
    assert _self.upsample is None and _self.downsample is None
    assert _self.norm1.num_groups == _self.norm2.num_groups
    assert temb is None

    # make sure norms are on cpu
    org_device = input_tensor.device
    cpu_device = torch.device("cpu")
    _self.norm1.to(cpu_device)
    _self.norm2.to(cpu_device)

    # GroupNormがCPUでfp16で動かない対策
    org_dtype = input_tensor.dtype
    if org_dtype == torch.float16:
        _self.norm1.to(torch.float32)
        _self.norm2.to(torch.float32)

    # すべてのテンソルをCPUに移動する
    input_tensor = input_tensor.to(cpu_device)
    hidden_states = input_tensor

    # どうもこれは結果が異なるようだ……
    # def sliced_norm1(norm, x):
    #     num_div = 4 if up_block_idx <= 2 else x.shape[1] // norm.num_groups
    #     sliced_tensor = torch.chunk(x, num_div, dim=1)
    #     sliced_weight = torch.chunk(norm.weight, num_div, dim=0)
    #     sliced_bias = torch.chunk(norm.bias, num_div, dim=0)
    #     print(sliced_tensor[0].shape, num_div, sliced_weight[0].shape, sliced_bias[0].shape)
    #     normed_tensor = []
    #     for i in range(num_div):
    #         n = torch.group_norm(sliced_tensor[i], norm.num_groups, sliced_weight[i], sliced_bias[i], norm.eps)
    #         normed_tensor.append(n)
    #         del n
    #     x = torch.cat(normed_tensor, dim=1)
    #     return num_div, x

    # normを分割すると結果が変わるので、ここだけは分割しない。GPUで計算するとVRAMが足りなくなるので、CPUで計算する。幸いCPUでもそこまで遅くない
    if org_dtype == torch.float16:
        hidden_states = hidden_states.to(torch.float32)
    hidden_states = _self.norm1(hidden_states)  # run on cpu
    if org_dtype == torch.float16:
        hidden_states = hidden_states.to(torch.float16)

    sliced = slice_h(hidden_states, num_slices)
    del hidden_states

    for i in range(len(sliced)):
        x = sliced[i]
        sliced[i] = None

        # 計算する部分だけGPUに移動する、以下同様
        x = x.to(org_device)
        x = _self.nonlinearity(x)
        x = _self.conv1(x)
        x = x.to(cpu_device)
        sliced[i] = x
        del x

    hidden_states = cat_h(sliced)
    del sliced

    if org_dtype == torch.float16:
        hidden_states = hidden_states.to(torch.float32)
    hidden_states = _self.norm2(hidden_states)  # run on cpu
    if org_dtype == torch.float16:
        hidden_states = hidden_states.to(torch.float16)

    sliced = slice_h(hidden_states, num_slices)
    del hidden_states

    for i in range(len(sliced)):
        x = sliced[i]
        sliced[i] = None

        x = x.to(org_device)
        x = _self.nonlinearity(x)
        x = _self.dropout(x)
        x = _self.conv2(x)
        x = x.to(cpu_device)
        sliced[i] = x
        del x

    hidden_states = cat_h(sliced)
    del sliced

    # make shortcut
    if _self.conv_shortcut is not None:
        sliced = list(torch.chunk(input_tensor, num_slices, dim=2))  # no padding in conv_shortcut パディングがないので普通にスライスする
        del input_tensor

        for i in range(len(sliced)):
            x = sliced[i]
            sliced[i] = None

            x = x.to(org_device)
            x = _self.conv_shortcut(x)
            x = x.to(cpu_device)
            sliced[i] = x
            del x

        input_tensor = torch.cat(sliced, dim=2)
        del sliced

    output_tensor = (input_tensor + hidden_states) / _self.output_scale_factor

    output_tensor = output_tensor.to(org_device)  # 次のレイヤーがGPUで計算する
    return output_tensor


class SlicingEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
        num_slices=2,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        self.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)  # とりあえずDiffusersのxformersを使う

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        # replace forward of ResBlocks
        def wrapper(func, module, num_slices):
            def forward(*args, **kwargs):
                return func(module, num_slices, *args, **kwargs)

            return forward

        self.num_slices = num_slices
        div = num_slices / (2 ** (len(self.down_blocks) - 1))  # 深い層はそこまで分割しなくていいので適宜減らす
        # print(f"initial divisor: {div}")
        if div >= 2:
            div = int(div)
            for resnet in self.mid_block.resnets:
                resnet.forward = wrapper(resblock_forward, resnet, div)
            # midblock doesn't have downsample

        for i, down_block in enumerate(self.down_blocks[::-1]):
            if div >= 2:
                div = int(div)
                # print(f"down block: {i} divisor: {div}")
                for resnet in down_block.resnets:
                    resnet.forward = wrapper(resblock_forward, resnet, div)
                if down_block.downsamplers is not None:
                    # print("has downsample")
                    for downsample in down_block.downsamplers:
                        downsample.forward = wrapper(self.downsample_forward, downsample, div * 2)
            div *= 2

    def forward(self, x):
        sample = x
        del x

        org_device = sample.device
        cpu_device = torch.device("cpu")

        # sample = self.conv_in(sample)
        sample = sample.to(cpu_device)
        sliced = slice_h(sample, self.num_slices)
        del sample

        for i in range(len(sliced)):
            x = sliced[i]
            sliced[i] = None

            x = x.to(org_device)
            x = self.conv_in(x)
            x = x.to(cpu_device)
            sliced[i] = x
            del x

        sample = cat_h(sliced)
        del sliced

        sample = sample.to(org_device)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        # ここも省メモリ化したいが、恐らくそこまでメモリを食わないので省略
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def downsample_forward(self, _self, num_slices, hidden_states):
        assert hidden_states.shape[1] == _self.channels
        assert _self.use_conv and _self.padding == 0
        print("downsample forward", num_slices, hidden_states.shape)

        org_device = hidden_states.device
        cpu_device = torch.device("cpu")

        hidden_states = hidden_states.to(cpu_device)
        pad = (0, 1, 0, 1)
        hidden_states = torch.nn.functional.pad(hidden_states, pad, mode="constant", value=0)

        # slice with even number because of stride 2
        # strideが2なので偶数でスライスする
        # slice with pad 1 both sides: to eliminate side effect of padding of conv2d
        size = (hidden_states.shape[2] + num_slices - 1) // num_slices
        size = size + 1 if size % 2 == 1 else size

        sliced = []
        for i in range(num_slices):
            if i == 0:
                sliced.append(hidden_states[:, :, : size + 1, :])
            else:
                end = size * (i + 1) + 1
                if hidden_states.shape[2] - end < 4:  # if the last slice is too small, use the rest of the tensor
                    end = hidden_states.shape[2]
                sliced.append(hidden_states[:, :, size * i - 1 : end, :])
                if end >= hidden_states.shape[2]:
                    break
        del hidden_states

        for i in range(len(sliced)):
            x = sliced[i]
            sliced[i] = None

            x = x.to(org_device)
            x = _self.conv(x)
            x = x.to(cpu_device)

            # ここだけ雰囲気が違うのはCopilotのせい
            if i == 0:
                hidden_states = x
            else:
                hidden_states = torch.cat([hidden_states, x], dim=2)

        hidden_states = hidden_states.to(org_device)
        # print("downsample forward done", hidden_states.shape)
        return hidden_states


class SlicingDecoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        num_slices=2,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        self.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)  # とりあえずDiffusersのxformersを使う

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # replace forward of ResBlocks
        def wrapper(func, module, num_slices):
            def forward(*args, **kwargs):
                return func(module, num_slices, *args, **kwargs)

            return forward

        self.num_slices = num_slices
        div = num_slices / (2 ** (len(self.up_blocks) - 1))
        print(f"initial divisor: {div}")
        if div >= 2:
            div = int(div)
            for resnet in self.mid_block.resnets:
                resnet.forward = wrapper(resblock_forward, resnet, div)
            # midblock doesn't have upsample

        for i, up_block in enumerate(self.up_blocks):
            if div >= 2:
                div = int(div)
                # print(f"up block: {i} divisor: {div}")
                for resnet in up_block.resnets:
                    resnet.forward = wrapper(resblock_forward, resnet, div)
                if up_block.upsamplers is not None:
                    # print("has upsample")
                    for upsample in up_block.upsamplers:
                        upsample.forward = wrapper(self.upsample_forward, upsample, div * 2)
            div *= 2

    def forward(self, z):
        sample = z
        del z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for i, up_block in enumerate(self.up_blocks):
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        # conv_out with slicing because of VRAM usage
        # conv_outはとてもVRAM使うのでスライスして対応
        org_device = sample.device
        cpu_device = torch.device("cpu")
        sample = sample.to(cpu_device)

        sliced = slice_h(sample, self.num_slices)
        del sample
        for i in range(len(sliced)):
            x = sliced[i]
            sliced[i] = None

            x = x.to(org_device)
            x = self.conv_out(x)
            x = x.to(cpu_device)
            sliced[i] = x
        sample = cat_h(sliced)
        del sliced

        sample = sample.to(org_device)
        return sample

    def upsample_forward(self, _self, num_slices, hidden_states, output_size=None):
        assert hidden_states.shape[1] == _self.channels
        assert _self.use_conv_transpose == False and _self.use_conv

        org_dtype = hidden_states.dtype
        org_device = hidden_states.device
        cpu_device = torch.device("cpu")

        hidden_states = hidden_states.to(cpu_device)
        sliced = slice_h(hidden_states, num_slices)
        del hidden_states

        for i in range(len(sliced)):
            x = sliced[i]
            sliced[i] = None

            x = x.to(org_device)

            # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
            # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
            # https://github.com/pytorch/pytorch/issues/86679
            # PyTorch 2で直らないかね……
            if org_dtype == torch.bfloat16:
                x = x.to(torch.float32)

            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

            if org_dtype == torch.bfloat16:
                x = x.to(org_dtype)

            x = _self.conv(x)

            # upsampleされてるのでpadは2になる
            if i == 0:
                x = x[:, :, :-2, :]
            elif i == num_slices - 1:
                x = x[:, :, 2:, :]
            else:
                x = x[:, :, 2:-2, :]

            x = x.to(cpu_device)
            sliced[i] = x
            del x

        hidden_states = torch.cat(sliced, dim=2)
        # print("us hidden_states", hidden_states.shape)
        del sliced

        hidden_states = hidden_states.to(org_device)
        return hidden_states


class SlicingAutoencoderKL(ModelMixin, ConfigMixin):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        num_slices: int = 16,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = SlicingEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            num_slices=num_slices,
        )

        # pass init params to Decoder
        self.decoder = SlicingDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            num_slices=num_slices,
        )

        self.quant_conv = torch.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)
        self.use_slicing = False

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # これはバッチ方向のスライシング　紛らわしい
    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
