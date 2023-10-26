# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import logging
import torch

from omegaconf import OmegaConf

from ldm.modules.karlo.kakao.models.clip import CustomizedCLIP, CustomizedTokenizer
from ldm.modules.karlo.kakao.models.prior_model import PriorDiffusionModel
from ldm.modules.karlo.kakao.models.decoder_model import Text2ImProgressiveModel
from ldm.modules.karlo.kakao.models.sr_64_256 import ImprovedSupRes64to256ProgressiveModel


SAMPLING_CONF = {
    "default": {
        "prior_sm": "25",
        "prior_n_samples": 1,
        "prior_cf_scale": 4.0,
        "decoder_sm": "50",
        "decoder_cf_scale": 8.0,
        "sr_sm": "7",
    },
    "fast": {
        "prior_sm": "25",
        "prior_n_samples": 1,
        "prior_cf_scale": 4.0,
        "decoder_sm": "25",
        "decoder_cf_scale": 8.0,
        "sr_sm": "7",
    },
}

CKPT_PATH = {
    "prior": "prior-ckpt-step=01000000-of-01000000.ckpt",
    "decoder": "decoder-ckpt-step=01000000-of-01000000.ckpt",
    "sr_256": "improved-sr-ckpt-step=1.2M.ckpt",
}


class BaseSampler:
    _PRIOR_CLASS = PriorDiffusionModel
    _DECODER_CLASS = Text2ImProgressiveModel
    _SR256_CLASS = ImprovedSupRes64to256ProgressiveModel

    def __init__(
        self,
        root_dir: str,
        sampling_type: str = "fast",
    ):
        self._root_dir = root_dir

        sampling_type = SAMPLING_CONF[sampling_type]
        self._prior_sm = sampling_type["prior_sm"]
        self._prior_n_samples = sampling_type["prior_n_samples"]
        self._prior_cf_scale = sampling_type["prior_cf_scale"]

        assert self._prior_n_samples == 1

        self._decoder_sm = sampling_type["decoder_sm"]
        self._decoder_cf_scale = sampling_type["decoder_cf_scale"]

        self._sr_sm = sampling_type["sr_sm"]

    def __repr__(self):
        line = ""
        line += f"Prior, sampling method: {self._prior_sm}, cf_scale: {self._prior_cf_scale}\n"
        line += f"Decoder, sampling method: {self._decoder_sm}, cf_scale: {self._decoder_cf_scale}\n"
        line += f"SR(64->256), sampling method: {self._sr_sm}"

        return line

    def load_clip(self, clip_path: str):
        clip = CustomizedCLIP.load_from_checkpoint(
            os.path.join(self._root_dir, clip_path)
        )
        clip = torch.jit.script(clip)
        clip.cuda()
        clip.eval()

        self._clip = clip
        self._tokenizer = CustomizedTokenizer()

    def load_prior(
        self,
        ckpt_path: str,
        clip_stat_path: str,
        prior_config: str = "configs/prior_1B_vit_l.yaml"
    ):
        logging.info(f"Loading prior: {ckpt_path}")

        config = OmegaConf.load(prior_config)
        clip_mean, clip_std = torch.load(
            os.path.join(self._root_dir, clip_stat_path), map_location="cpu"
        )

        prior = self._PRIOR_CLASS.load_from_checkpoint(
            config,
            self._tokenizer,
            clip_mean,
            clip_std,
            os.path.join(self._root_dir, ckpt_path),
            strict=True,
        )
        prior.cuda()
        prior.eval()
        logging.info("done.")

        self._prior = prior

    def load_decoder(self, ckpt_path: str, decoder_config: str = "configs/decoder_900M_vit_l.yaml"):
        logging.info(f"Loading decoder: {ckpt_path}")

        config = OmegaConf.load(decoder_config)
        decoder = self._DECODER_CLASS.load_from_checkpoint(
            config,
            self._tokenizer,
            os.path.join(self._root_dir, ckpt_path),
            strict=True,
        )
        decoder.cuda()
        decoder.eval()
        logging.info("done.")

        self._decoder = decoder

    def load_sr_64_256(self, ckpt_path: str, sr_config: str = "configs/improved_sr_64_256_1.4B.yaml"):
        logging.info(f"Loading SR(64->256): {ckpt_path}")

        config = OmegaConf.load(sr_config)
        sr = self._SR256_CLASS.load_from_checkpoint(
            config, os.path.join(self._root_dir, ckpt_path), strict=True
        )
        sr.cuda()
        sr.eval()
        logging.info("done.")

        self._sr_64_256 = sr