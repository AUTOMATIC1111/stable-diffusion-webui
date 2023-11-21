#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/21 9:51 AM
# @Author  : wangdongming
# @Site    : 
# @File    : lcm.py
# @Software: xingzhe.ai
import torch
import tqdm
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from modules.script_callbacks import on_before_ui

NAME = 'LCM-Alpha'
ALIAS = 'lcm'

from modules import scripts


class ExtraLCMSampler(scripts.Script):

    def title(self):
        return "LCM-sampler"

    def show(self, is_img2img):
        return False


@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = k_diffusion.sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in tqdm.auto.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
    return x


def add_lcm_sampler():
    if NAME not in [x.name for x in sd_samplers.all_samplers]:
        lcm_samplers = [(NAME, sample_lcm, [ALIAS], {})]
        samplers_data_lcm_samplers = [
            sd_samplers_common.SamplerData(label,
                                           lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(
                                               funcname, model), aliases, options)
            for label, funcname, aliases, options in lcm_samplers
            if callable(funcname) or (isinstance(funcname, str) and hasattr(k_diffusion.sampling, funcname))
        ]
        sd_samplers.all_samplers += samplers_data_lcm_samplers
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()


on_before_ui(add_lcm_sampler)

print("imported lcm sampler!!!")

