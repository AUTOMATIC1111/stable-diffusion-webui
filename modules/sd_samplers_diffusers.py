from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from modules import sd_samplers_common

samplers_data_diffusers = [
    sd_samplers_common.SamplerData('UniPC', lambda model: DiffusionSampler('UniPC', UniPCMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDIM', lambda model: DiffusionSampler('DDIM', DDIMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDPM', lambda model: DiffusionSampler('DDPM', DDPMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DEIS', lambda model: DiffusionSampler('DEIS', DEISMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM++ 2M', lambda model: DiffusionSampler('DPM++ 2M', DPMSolverMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM++ 1S', lambda model: DiffusionSampler('DPM++ 1S', DPMSolverSinglestepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE', lambda model: DiffusionSampler('DPM++ 2M SDE',  DPMSolverMultistepScheduler, model, algorithm_type="sde-dpmsolver++"), [], {}),
    sd_samplers_common.SamplerData('DPM++ 2M Karras', lambda model: DiffusionSampler('DPM++ 2M Karras', DPMSolverMultistepScheduler, model, use_karras_sigmas=True), [], {}),
    sd_samplers_common.SamplerData('DPM++ 1S Karras', lambda model: DiffusionSampler('DPM++ 1S Karras', DPMSolverSinglestepScheduler, model, use_karras_sigmas=True), [], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE Karras', lambda model: DiffusionSampler('DPM++ 2M SDE Karras', DPMSolverMultistepScheduler, model, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"), [], {}),
    sd_samplers_common.SamplerData('Euler', lambda model: DiffusionSampler('Euler', EulerDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler a', lambda model: DiffusionSampler('Euler a', EulerAncestralDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Heun', lambda model: DiffusionSampler('Heun', HeunDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPM2++ 2M', lambda model: DiffusionSampler('KDPM2', KDPM2DiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('PNDM', lambda model: DiffusionSampler('PNDM', PNDMScheduler, model), [], {}),
]

class DiffusionSampler:
    def __init__(self, name, constructor, sd_model, **kwargs):
        self.sampler = constructor.from_pretrained(sd_model, subfolder="scheduler", **kwargs)
        self.sampler.name = name
