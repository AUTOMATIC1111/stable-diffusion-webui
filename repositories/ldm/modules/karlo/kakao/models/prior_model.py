# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import copy
import torch

from ldm.modules.karlo.kakao.modules import create_gaussian_diffusion
from ldm.modules.karlo.kakao.modules.xf import PriorTransformer


class PriorDiffusionModel(torch.nn.Module):
    """
    A prior that generates clip image feature based on the text prompt.

    :param config: yaml config to define the decoder.
    :param tokenizer: tokenizer used in clip.
    :param clip_mean: mean to normalize the clip image feature (zero-mean, unit variance).
    :param clip_std: std to noramlize the clip image feature (zero-mean, unit variance).
    """

    def __init__(self, config, tokenizer, clip_mean, clip_std):
        super().__init__()

        self._conf = config
        self._model_conf = config.model.hparams
        self._diffusion_kwargs = dict(
            steps=config.diffusion.steps,
            learn_sigma=config.diffusion.learn_sigma,
            sigma_small=config.diffusion.sigma_small,
            noise_schedule=config.diffusion.noise_schedule,
            use_kl=config.diffusion.use_kl,
            predict_xstart=config.diffusion.predict_xstart,
            rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
            timestep_respacing=config.diffusion.timestep_respacing,
        )
        self._tokenizer = tokenizer

        self.register_buffer("clip_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("clip_std", clip_std[None, :], persistent=False)

        causal_mask = self.get_causal_mask()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.model = PriorTransformer(
            text_ctx=self._model_conf.text_ctx,
            xf_width=self._model_conf.xf_width,
            xf_layers=self._model_conf.xf_layers,
            xf_heads=self._model_conf.xf_heads,
            xf_final_ln=self._model_conf.xf_final_ln,
            clip_dim=self._model_conf.clip_dim,
        )

        cf_token, cf_mask = self.set_cf_text_tensor()
        self.register_buffer("cf_token", cf_token, persistent=False)
        self.register_buffer("cf_mask", cf_mask, persistent=False)

    @classmethod
    def load_from_checkpoint(
        cls, config, tokenizer, clip_mean, clip_std, ckpt_path, strict: bool = True
    ):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        model = cls(config, tokenizer, clip_mean, clip_std)
        model.load_state_dict(ckpt, strict=strict)
        return model

    def set_cf_text_tensor(self):
        return self._tokenizer.padded_tokens_and_mask([""], self.model.text_ctx)

    def get_sample_fn(self, timestep_respacing):
        use_ddim = timestep_respacing.startswith(("ddim", "fast"))

        diffusion_kwargs = copy.deepcopy(self._diffusion_kwargs)
        diffusion_kwargs.update(timestep_respacing=timestep_respacing)
        diffusion = create_gaussian_diffusion(**diffusion_kwargs)
        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop

        return sample_fn

    def get_causal_mask(self):
        seq_len = self._model_conf.text_ctx + 4
        mask = torch.empty(seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        mask = mask[None, ...]
        return mask

    def forward(
        self,
        txt_feat,
        txt_feat_seq,
        mask,
        cf_guidance_scales=None,
        timestep_respacing=None,
        denoised_fn=True,
    ):
        # cfg should be enabled in inference
        assert cf_guidance_scales is not None and all(cf_guidance_scales > 0.0)

        bsz_ = txt_feat.shape[0]
        bsz = bsz_ // 2

        def guided_model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = (
                model_out[:, : int(x_t.shape[1])],
                model_out[:, int(x_t.shape[1]) :],
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cf_guidance_scales.view(-1, 1) * (
                cond_eps - uncond_eps
            )
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        cond = {
            "text_emb": txt_feat,
            "text_enc": txt_feat_seq,
            "mask": mask,
            "causal_mask": self.causal_mask,
        }
        sample_fn = self.get_sample_fn(timestep_respacing)
        sample = sample_fn(
            guided_model_fn,
            (bsz_, self.model.clip_dim),
            noise=None,
            device=txt_feat.device,
            clip_denoised=False,
            denoised_fn=lambda x: torch.clamp(x, -10, 10),
            model_kwargs=cond,
        )
        sample = (sample * self.clip_std) + self.clip_mean

        return sample[:bsz]
