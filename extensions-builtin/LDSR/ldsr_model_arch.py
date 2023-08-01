import os
import gc
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
import safetensors.torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, ismap
from modules import shared, sd_hijack, devices

cached_ldsr_model: torch.nn.Module = None


# Create LDSR Class
class LDSR:
    def load_model_from_config(self, half_attention):
        global cached_ldsr_model

        if shared.opts.ldsr_cached and cached_ldsr_model is not None:
            print("Loading model from cache")
            model: torch.nn.Module = cached_ldsr_model
        else:
            print(f"Loading model from {self.modelPath}")
            _, extension = os.path.splitext(self.modelPath)
            if extension.lower() == ".safetensors":
                pl_sd = safetensors.torch.load_file(self.modelPath, device="cpu")
            else:
                pl_sd = torch.load(self.modelPath, map_location="cpu")
            sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
            config = OmegaConf.load(self.yamlPath)
            config.model.target = "ldm.models.diffusion.ddpm.LatentDiffusionV1"
            model: torch.nn.Module = instantiate_from_config(config.model)
            model.load_state_dict(sd, strict=False)
            model = model.to(shared.device)
            if half_attention:
                model = model.half()
            if shared.cmd_opts.opt_channelslast:
                model = model.to(memory_format=torch.channels_last)

            sd_hijack.model_hijack.hijack(model) # apply optimization
            model.eval()

            if shared.opts.ldsr_cached:
                cached_ldsr_model = model

        return {"model": model}

    def __init__(self, model_path, yaml_path):
        self.modelPath = model_path
        self.yamlPath = yaml_path

    @staticmethod
    def run(model, selected_path, custom_steps, eta):
        example = get_cond(selected_path)

        n_runs = 1
        guider = None
        ckwargs = None
        ddim_use_x0_pred = False
        temperature = 1.
        eta = eta
        custom_shape = None

        height, width = example["image"].shape[1:3]
        split_input = height >= 128 and width >= 128

        if split_input:
            ks = 128
            stride = 64
            vqf = 4  #
            model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                        "vqf": vqf,
                                        "patch_distributed_vq": True,
                                        "tie_braker": False,
                                        "clip_max_weight": 0.5,
                                        "clip_min_weight": 0.01,
                                        "clip_max_tie_weight": 0.5,
                                        "clip_min_tie_weight": 0.01}
        else:
            if hasattr(model, "split_input_params"):
                delattr(model, "split_input_params")

        x_t = None
        logs = None
        for _ in range(n_runs):
            if custom_shape is not None:
                x_t = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
                x_t = repeat(x_t, '1 c h w -> b c h w', b=custom_shape[0])

            logs = make_convolutional_sample(example, model,
                                             custom_steps=custom_steps,
                                             eta=eta, quantize_x0=False,
                                             custom_shape=custom_shape,
                                             temperature=temperature, noise_dropout=0.,
                                             corrector=guider, corrector_kwargs=ckwargs, x_T=x_t,
                                             ddim_use_x0_pred=ddim_use_x0_pred
                                             )
        return logs

    def super_resolution(self, image, steps=100, target_scale=2, half_attention=False):
        model = self.load_model_from_config(half_attention)

        # Run settings
        diffusion_steps = int(steps)
        eta = 1.0


        gc.collect()
        devices.torch_gc()

        im_og = image
        width_og, height_og = im_og.size
        # If we can adjust the max upscale size, then the 4 below should be our variable
        down_sample_rate = target_scale / 4
        wd = width_og * down_sample_rate
        hd = height_og * down_sample_rate
        width_downsampled_pre = int(np.ceil(wd))
        height_downsampled_pre = int(np.ceil(hd))

        if down_sample_rate != 1:
            print(
                f'Downsampling from [{width_og}, {height_og}] to [{width_downsampled_pre}, {height_downsampled_pre}]')
            im_og = im_og.resize((width_downsampled_pre, height_downsampled_pre), Image.LANCZOS)
        else:
            print(f"Down sample rate is 1 from {target_scale} / 4 (Not downsampling)")

        # pad width and height to multiples of 64, pads with the edge values of image to avoid artifacts
        pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(im_og.size) / 64).astype(int)), axis=0) * 64 - im_og.size
        im_padded = Image.fromarray(np.pad(np.array(im_og), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))

        logs = self.run(model["model"], im_padded, diffusion_steps, eta)

        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        a = Image.fromarray(sample[0])

        # remove padding
        a = a.crop((0, 0) + tuple(np.array(im_og.size) * 4))

        del model
        gc.collect()
        devices.torch_gc()

        return a


def get_cond(selected_path):
    example = {}
    up_f = 4
    c = selected_path.convert('RGB')
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]],
                                                    antialias=True)
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')
    c = rearrange(c, '1 c h w -> 1 h w c')
    c = 2. * c - 1.

    c = c.to(shared.device)
    example["LR_image"] = c
    example["image"] = c_up

    return example


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, temperature=1., score_corrector=None,
                    corrector_kwargs=None, x_t=None
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_t=x_t)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, custom_steps=None, eta=1.0, quantize_x0=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, ddim_use_x0_pred=False):
    log = {}

    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key == 'class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, mask=None, x0=z0,
                                                temperature=temperature, score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_t=x_T)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except Exception:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log
