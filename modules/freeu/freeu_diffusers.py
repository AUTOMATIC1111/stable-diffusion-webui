# https://github.com/lyn-rgb/FreeU_Diffusers/blob/diffusers-v0.21.2/free_lunch_utils.py

"""
register_free_upblock2d(pipe)
register_free_crossattn_upblock2d(pipe)
"""


from typing import Any, Dict, Optional, Tuple
import torch
import torch.fft as fft
from diffusers.utils import is_torch_version


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    B, C, H, W = x.shape
    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).to(x.device)

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered


def register_upblock2d(model):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, scale: float = 1.0):
            for resnet in self.resnets:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb, scale=scale)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

            return hidden_states

        return forward

    for _i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)


def register_free_upblock2d(model, b1=1.2, b2=1.4, s1=0.9, s2=0.2):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, scale: float = 1.0):
            for resnet in self.resnets:
                # pop res hidden states
                #print(f"in free upblock2d, hidden states shape: {hidden_states.shape}")
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # --------------- FreeU code -----------------------
                # Only operate on the first two stages
                if hidden_states.shape[1] == 1280:
                    hidden_states[:,:640] = hidden_states[:,:640] * self.b1
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s1)
                if hidden_states.shape[1] == 640:
                    hidden_states[:,:320] = hidden_states[:,:320] * self.b2
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s2)
                # ---------------------------------------------------------

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb, scale=scale)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

            return hidden_states

        return forward

    for _i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            upsample_block.b1 = b1
            upsample_block.b2 = b2
            upsample_block.s1 = s1
            upsample_block.s2 = s2


def register_crossattn_upblock2d(model):
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)

            return hidden_states

        return forward

    for _i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)


def register_free_crossattn_upblock2d(model, b1=1.2, b2=1.4, s1=0.9, s2=0.2):
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                #print(f"in free crossatten upblock2d, hidden states shape: {hidden_states.shape}")
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # --------------- FreeU code -----------------------
                # Only operate on the first two stages
                if hidden_states.shape[1] == 1280:
                    hidden_states[:,:640] = hidden_states[:,:640] * self.b1
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s1)
                if hidden_states.shape[1] == 640:
                    hidden_states[:,:320] = hidden_states[:,:320] * self.b2
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=self.s2)
                # ---------------------------------------------------------

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)

            return hidden_states

        return forward

    for _i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            upsample_block.b1 = b1
            upsample_block.b2 = b2
            upsample_block.s1 = s1
            upsample_block.s2 = s2
