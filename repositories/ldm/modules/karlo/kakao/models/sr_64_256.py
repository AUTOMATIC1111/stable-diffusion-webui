# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import copy
import torch

from ldm.modules.karlo.kakao.modules.unet import SuperResUNetModel
from ldm.modules.karlo.kakao.modules import create_gaussian_diffusion


class ImprovedSupRes64to256ProgressiveModel(torch.nn.Module):
    """
    ImprovedSR model fine-tunes the pretrained DDPM-based SR model by using adversarial and perceptual losses.
    In specific, the low-resolution sample is iteratively recovered by 6 steps with the frozen pretrained SR model.
    In the following additional one step, a seperate fine-tuned model recovers high-frequency details.
    This approach greatly improves the fidelity of images of 256x256px, even with small number of reverse steps.
    """

    def __init__(self, config):
        super().__init__()

        self._config = config
        self._diffusion_kwargs = dict(
            steps=config.diffusion.steps,
            learn_sigma=config.diffusion.learn_sigma,
            sigma_small=config.diffusion.sigma_small,
            noise_schedule=config.diffusion.noise_schedule,
            use_kl=config.diffusion.use_kl,
            predict_xstart=config.diffusion.predict_xstart,
            rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
        )

        self.model_first_steps = SuperResUNetModel(
            in_channels=3,  # auto-changed to 6 inside the model
            model_channels=config.model.hparams.channels,
            out_channels=3,
            num_res_blocks=config.model.hparams.depth,
            attention_resolutions=(),  # no attention
            dropout=config.model.hparams.dropout,
            channel_mult=config.model.hparams.channels_multiple,
            resblock_updown=True,
            use_middle_attention=False,
        )
        self.model_last_step = SuperResUNetModel(
            in_channels=3,  # auto-changed to 6 inside the model
            model_channels=config.model.hparams.channels,
            out_channels=3,
            num_res_blocks=config.model.hparams.depth,
            attention_resolutions=(),  # no attention
            dropout=config.model.hparams.dropout,
            channel_mult=config.model.hparams.channels_multiple,
            resblock_updown=True,
            use_middle_attention=False,
        )

    @classmethod
    def load_from_checkpoint(cls, config, ckpt_path, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        model = cls(config)
        model.load_state_dict(ckpt, strict=strict)
        return model

    def get_sample_fn(self, timestep_respacing):
        diffusion_kwargs = copy.deepcopy(self._diffusion_kwargs)
        diffusion_kwargs.update(timestep_respacing=timestep_respacing)
        diffusion = create_gaussian_diffusion(**diffusion_kwargs)
        return diffusion.p_sample_loop_progressive_for_improved_sr

    def forward(self, low_res, timestep_respacing="7", **kwargs):
        assert (
            timestep_respacing == "7"
        ), "different respacing method may work, but no guaranteed"

        sample_fn = self.get_sample_fn(timestep_respacing)
        sample_outputs = sample_fn(
            self.model_first_steps,
            self.model_last_step,
            shape=low_res.shape,
            clip_denoised=True,
            model_kwargs=dict(low_res=low_res),
            **kwargs,
        )
        for x in sample_outputs:
            sample = x["sample"]
            yield sample
