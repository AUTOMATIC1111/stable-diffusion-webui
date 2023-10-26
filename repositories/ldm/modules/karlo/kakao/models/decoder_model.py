# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import copy
import torch

from ldm.modules.karlo.kakao.modules import create_gaussian_diffusion
from ldm.modules.karlo.kakao.modules.unet import PLMImUNet


class Text2ImProgressiveModel(torch.nn.Module):
    """
    A decoder that generates 64x64px images based on the text prompt.

    :param config: yaml config to define the decoder.
    :param tokenizer: tokenizer used in clip.
    """

    def __init__(
        self,
        config,
        tokenizer,
    ):
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

        self.model = self.create_plm_dec_model()

        cf_token, cf_mask = self.set_cf_text_tensor()
        self.register_buffer("cf_token", cf_token, persistent=False)
        self.register_buffer("cf_mask", cf_mask, persistent=False)

    @classmethod
    def load_from_checkpoint(cls, config, tokenizer, ckpt_path, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        model = cls(config, tokenizer)
        model.load_state_dict(ckpt, strict=strict)
        return model

    def create_plm_dec_model(self):
        image_size = self._model_conf.image_size
        if self._model_conf.channel_mult == "":
            if image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = tuple(
                int(ch_mult) for ch_mult in self._model_conf.channel_mult.split(",")
            )
            assert 2 ** (len(channel_mult) + 2) == image_size

        attention_ds = []
        for res in self._model_conf.attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return PLMImUNet(
            text_ctx=self._model_conf.text_ctx,
            xf_width=self._model_conf.xf_width,
            in_channels=3,
            model_channels=self._model_conf.num_channels,
            out_channels=6 if self._model_conf.learn_sigma else 3,
            num_res_blocks=self._model_conf.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=self._model_conf.dropout,
            channel_mult=channel_mult,
            num_heads=self._model_conf.num_heads,
            num_head_channels=self._model_conf.num_head_channels,
            num_heads_upsample=self._model_conf.num_heads_upsample,
            use_scale_shift_norm=self._model_conf.use_scale_shift_norm,
            resblock_updown=self._model_conf.resblock_updown,
            clip_dim=self._model_conf.clip_dim,
            clip_emb_mult=self._model_conf.clip_emb_mult,
            clip_emb_type=self._model_conf.clip_emb_type,
            clip_emb_drop=self._model_conf.clip_emb_drop,
        )

    def set_cf_text_tensor(self):
        return self._tokenizer.padded_tokens_and_mask([""], self.model.text_ctx)

    def get_sample_fn(self, timestep_respacing):
        use_ddim = timestep_respacing.startswith(("ddim", "fast"))

        diffusion_kwargs = copy.deepcopy(self._diffusion_kwargs)
        diffusion_kwargs.update(timestep_respacing=timestep_respacing)
        diffusion = create_gaussian_diffusion(**diffusion_kwargs)
        sample_fn = (
            diffusion.ddim_sample_loop_progressive
            if use_ddim
            else diffusion.p_sample_loop_progressive
        )

        return sample_fn

    def forward(
        self,
        txt_feat,
        txt_feat_seq,
        tok,
        mask,
        img_feat=None,
        cf_guidance_scales=None,
        timestep_respacing=None,
    ):
        # cfg should be enabled in inference
        assert cf_guidance_scales is not None and all(cf_guidance_scales > 0.0)
        assert img_feat is not None

        bsz = txt_feat.shape[0]
        img_sz = self._model_conf.image_size

        def guided_model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cf_guidance_scales.view(-1, 1, 1, 1) * (
                cond_eps - uncond_eps
            )
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        cf_feat = self.model.cf_param.unsqueeze(0)
        cf_feat = cf_feat.expand(bsz // 2, -1)
        feat = torch.cat([img_feat, cf_feat.to(txt_feat.device)], dim=0)

        cond = {
            "y": feat,
            "txt_feat": txt_feat,
            "txt_feat_seq": txt_feat_seq,
            "mask": mask,
        }
        sample_fn = self.get_sample_fn(timestep_respacing)
        sample_outputs = sample_fn(
            guided_model_fn,
            (bsz, 3, img_sz, img_sz),
            noise=None,
            device=txt_feat.device,
            clip_denoised=True,
            model_kwargs=cond,
        )

        for out in sample_outputs:
            sample = out["sample"]
            yield sample if cf_guidance_scales is None else sample[
                : sample.shape[0] // 2
            ]


class Text2ImModel(Text2ImProgressiveModel):
    def forward(
        self,
        txt_feat,
        txt_feat_seq,
        tok,
        mask,
        img_feat=None,
        cf_guidance_scales=None,
        timestep_respacing=None,
    ):
        last_out = None
        for out in super().forward(
            txt_feat,
            txt_feat_seq,
            tok,
            mask,
            img_feat,
            cf_guidance_scales,
            timestep_respacing,
        ):
            last_out = out
        return last_out
