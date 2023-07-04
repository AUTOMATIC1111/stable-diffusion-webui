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

config = {
    'All': { 'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end': 0.02, 'beta_schedule': 'linear', 'prediction_type': 'epsilon' },
    'UniPC': { 'solver_order': 2, 'thresholding': False, 'dynamic_thresholding_ratio': 0.995, 'sample_max_value': 1.0, 'predict_x0': 'bh2', 'lower_order_final': True },
    'DDIM': { 'clip_sample': True, 'set_alpha_to_one': True, 'steps_offset': 0, 'thresholding': False, 'dynamic_thresholding_ratio': 0.995, 'clip_sample_range': 1.0, 'sample_max_value': 1.0, 'timestep_spacing': 'leading', 'rescale_betas_zero_snr': False },
    'DEIS': { 'solver_order': 2, 'thresholding': False, 'dynamic_thresholding_ratio': 0.995, 'sample_max_value': 1.0, 'algorithm_type': "deis", 'solver_type': "logrho", 'lower_order_final': True },
    'Euler a': {},
}

samplers_data_diffusers = [
    sd_samplers_common.SamplerData('UniPC', lambda model: DiffusionSampler('UniPC', UniPCMultistepScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DDIM', lambda model: DiffusionSampler('DDIM', DDIMScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('DDPM', lambda model: DiffusionSampler('DDPM', DDPMScheduler, model), [], {}),
    sd_samplers_common.SamplerData('DEIS', lambda model: DiffusionSampler('DEIS', DEISMultistepScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M', lambda model: DiffusionSampler('DPM++ 2M', DPMSolverMultistepScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('DPM++ 1S', lambda model: DiffusionSampler('DPM++ 1S', DPMSolverSinglestepScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE', lambda model: DiffusionSampler('DPM++ 2M SDE', DPMSolverMultistepScheduler, model, algorithm_type="sde-dpmsolver++"), [], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M Karras', lambda model: DiffusionSampler('DPM++ 2M Karras', DPMSolverMultistepScheduler, model, use_karras_sigmas=True), [], {}),
    # sd_samplers_common.SamplerData('DPM++ 1S Karras', lambda model: DiffusionSampler('DPM++ 1S Karras', DPMSolverSinglestepScheduler, model, use_karras_sigmas=True), [], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE Karras', lambda model: DiffusionSampler('DPM++ 2M SDE Karras', DPMSolverMultistepScheduler, model, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"), [], {}),
    # sd_samplers_common.SamplerData('Euler', lambda model: DiffusionSampler('Euler', EulerDiscreteScheduler, model), [], {}),
    sd_samplers_common.SamplerData('Euler a', lambda model: DiffusionSampler('Euler a', EulerAncestralDiscreteScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('Heun', lambda model: DiffusionSampler('Heun', HeunDiscreteScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('DPM2++ 2M', lambda model: DiffusionSampler('KDPM2', KDPM2DiscreteScheduler, model), [], {}),
    # sd_samplers_common.SamplerData('PNDM', lambda model: DiffusionSampler('PNDM', PNDMScheduler, model), [], {}),
]

class DiffusionSampler:
    def __init__(self, name, constructor, model, **kwargs):
        self.config = config['All'].copy()
        for key, value in config.get(name, {}).items(): # diffusers defaults
            if key in self.config:
                self.config[key] = value
        for key, value in model.scheduler.config.items(): # model defaults
            if key in self.config:
                self.config[key] = value
        for key, value in kwargs.items(): # user args
            if key in self.config:
                self.config[key] = value
        self.sampler = constructor(**self.config)
        self.sampler.name = name
