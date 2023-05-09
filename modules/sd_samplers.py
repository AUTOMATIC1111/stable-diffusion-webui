from modules import sd_samplers_compvis, sd_samplers_kdiffusion, shared

# imports for functions that previously were here and are used by other modules
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image

all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_compvis.samplers_data_compvis,
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers = []
samplers_for_img2img = []
samplers_map = {}


def create_sampler(name, model):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    assert config is not None, f'bad sampler name: {name}'

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers():
    global samplers, samplers_for_img2img

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


set_samplers()
