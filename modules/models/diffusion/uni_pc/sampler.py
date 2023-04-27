"""SAMPLING ONLY."""

import torch

from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC
from modules import shared, devices
from ldm.modules.diffusionmodules.util import extract_into_tensor


class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.before_sample = None
        self.after_sample = None
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        # persist steps so we can eventually find denoising strength
        self.inflated_steps = ddim_num_steps

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        # first time we have all the info to get the real parameters from the ui
        hires_steps = t[0] + 1
        inflated_steps = self.inflated_steps
        self.denoising_strength = hires_steps/inflated_steps

        adjusted_steps = int(hires_steps * self.denoising_strength)
        self.steps = max(adjusted_steps, shared.opts.uni_pc_order+1)

        t = torch.full(t.shape, self.steps).to(t.device)

        timesteps = torch.asarray(list(range(
            t,
            self.model.num_timesteps,
            self.model.num_timesteps // hires_steps,
        ))) + 1
        alphas = self.model.alphas_cumprod[timesteps]
        sqrt_one_minus_alphas = torch.sqrt(1. - alphas)
        a = extract_into_tensor(torch.sqrt(alphas), t, x0.shape) * x0 
        b = extract_into_tensor(sqrt_one_minus_alphas, t, x0.shape) * noise

        return (a+b)

    def decode(self, x_latent, conditioning, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):
        #print(f'steps {self.steps} denoising {self.denoising_strength}')

        noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        # same as in .sample(), i guess
        model_type = "v" if self.model.parameterization == "v" else "noise"

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            noise_schedule,
            model_type=model_type,
            guidance_type="classifier-free",
            #condition=conditioning,
            #unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        self.uni_pc = UniPC(
                model_fn,
                noise_schedule,
                predict_x0=True,
                thresholding=False,
                variant=shared.opts.uni_pc_variant,
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                before_sample=self.before_sample,
                after_sample=self.after_sample,
                after_update=self.after_update,
            )

        return self.uni_pc.sample(
                x_latent,
                steps=self.steps,
                skip_type=shared.opts.uni_pc_skip_type,
                method="multistep",
                order=shared.opts.uni_pc_order,
                lower_order_final=shared.opts.uni_pc_lower_order_final,
                t_start=self.denoising_strength,
            )

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != devices.device:
                attr = attr.to(devices.device)
        setattr(self, name, attr)

    def set_hooks(self, before_sample, after_sample, after_update):
        self.before_sample = before_sample
        self.after_sample = after_sample
        self.after_update = after_update

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for UniPC sampling is {size}')

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        # SD 1.X is "noise", SD 2.X is "v"
        model_type = "v" if self.model.parameterization == "v" else "noise"

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type=model_type,
            guidance_type="classifier-free",
            #condition=conditioning,
            #unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant=shared.opts.uni_pc_variant, condition=conditioning, unconditional_condition=unconditional_conditioning, before_sample=self.before_sample, after_sample=self.after_sample, after_update=self.after_update)
        x = uni_pc.sample(img, steps=S, skip_type=shared.opts.uni_pc_skip_type, method="multistep", order=shared.opts.uni_pc_order, lower_order_final=shared.opts.uni_pc_lower_order_final)

        return x.to(device), None
