from modules import sd_samplers_compvis, sd_samplers_kdiffusion, sd_samplers_diffusors, shared
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image # pylint: disable=unused-import
from modules.shared import backend, Backend

if backend == Backend.ORIGINAL:
    all_samplers = [
        *sd_samplers_kdiffusion.samplers_data_k_diffusion,
        *sd_samplers_compvis.samplers_data_compvis,
    ]
else:
    all_samplers = [
        *sd_samplers_diffusors.samplers_data_diffusors,
    ]
all_samplers_map = {x.name: x for x in all_samplers}
samplers = all_samplers
samplers_for_img2img = all_samplers
samplers_map = {}


def find_sampler_config(name):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]
    return config


def create_sampler(name, model):
    config = find_sampler_config(name)
    assert config is not None, f'bad sampler name: {name}'
    if backend == Backend.ORIGINAL:
        sampler = config.constructor(model)
        sampler.config = config
        return sampler
    else:
        sampler = config.constructor(model.sd_checkpoint_info.filename)
        model.scheduler = sampler.sampler
        return sampler.sampler


def set_samplers():
    global samplers, samplers_for_img2img # pylint: disable=global-statement
    shown_img2img = set(shared.opts.show_samplers)
    if len(shared.opts.show_samplers) == 0:
        shown = {'PLMS', 'UniPC'}
    else:
        shown = set(shared.opts.show_samplers + ['PLMS'])
    samplers = [x for x in all_samplers if x.name in shown]
    samplers_for_img2img = [x for x in all_samplers if x.name in shown_img2img]
    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name
