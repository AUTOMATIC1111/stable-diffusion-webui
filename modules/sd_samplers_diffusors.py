from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    # KarrasVeScheduler,
    # RePaintScheduler,
    # ScoreSdeVeScheduler,
    # UnCLIPScheduler,
    # VQDiffusionScheduler,
)
from modules import sd_samplers_common
    # scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(shared.cmd_opts.ckpt, subfolder="scheduler")

samplers_data_diffusors = [
    sd_samplers_common.SamplerData('UniPC', lambda model: DiffusionSampler('UniPC', UniPCMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDIM', lambda model: DiffusionSampler('DDIM', DDIMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDPMS', lambda model: DiffusionSampler('DDPMS', DDPMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DEIS', lambda model: DiffusionSampler('DEIS', DEISMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DPMSolver', lambda model: DiffusionSampler('DPMSolver', DPMSolverMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler', lambda model: DiffusionSampler('Euler', EulerDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('EulerAncestral', lambda model: DiffusionSampler('EulerAncestral', EulerAncestralDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Heun', lambda model: DiffusionSampler('Heun', HeunDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('IPNDM', lambda model: DiffusionSampler('IPNDM', IPNDMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('KDPM2Ancestral', lambda model: DiffusionSampler('KDPM2Ancestral', KDPM2AncestralDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('PNDMS', lambda model: DiffusionSampler('PNDMS', PNDMScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('KarrasVe', lambda model: DiffusionSampler('KarrasVe', KarrasVeScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('RePaint', lambda model: DiffusionSampler('RePaint', RePaintScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('ScoreSdeVe', lambda model: DiffusionSampler('ScoreSdeVe', ScoreSdeVeScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('UnCLIP', lambda model: DiffusionSampler('UnCLIP', UnCLIPScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('VQDiffusion', lambda model: DiffusionSampler('VQDiffusion', VQDiffusionScheduler, model), [], {}),
]


class DiffusionSampler:
    def __init__(self, name, constructor, sd_model):
        self.sampler = constructor.from_pretrained(sd_model, subfolder="scheduler")
        self.sampler.name = name
