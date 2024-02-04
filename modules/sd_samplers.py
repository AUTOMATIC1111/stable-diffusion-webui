import os
from modules import shared
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image # pylint: disable=unused-import

debug = shared.log.trace if os.environ.get('SD_SAMPLER_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: SAMPLER')
all_samplers = []
all_samplers = []
all_samplers_map = {}
samplers = all_samplers
samplers_for_img2img = all_samplers
samplers_map = {}
loaded_config = None
loaded_sampler = None


def list_samplers(backend_name = shared.backend):
    global all_samplers # pylint: disable=global-statement
    global all_samplers_map # pylint: disable=global-statement
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    global samplers_map # pylint: disable=global-statement
    if backend_name == shared.Backend.ORIGINAL:
        from modules import sd_samplers_compvis, sd_samplers_kdiffusion
        all_samplers = [*sd_samplers_compvis.samplers_data_compvis, *sd_samplers_kdiffusion.samplers_data_k_diffusion]
    else:
        from modules import sd_samplers_diffusers
        all_samplers = [*sd_samplers_diffusers.samplers_data_diffusers]
    all_samplers_map = {x.name: x for x in all_samplers}
    samplers = all_samplers
    samplers_for_img2img = all_samplers
    samplers_map = {}
    # shared.log.debug(f'Available samplers: {[x.name for x in all_samplers]}')


def find_sampler_config(name):
    if name is not None and name != 'None':
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]
    return config


def visible_sampler_names():
    visible_samplers = [x for x in all_samplers if x.name in shared.opts.show_samplers] if len(shared.opts.show_samplers) > 0 else all_samplers
    return visible_samplers


def create_sampler(name, model):
    global loaded_config, loaded_sampler # pylint: disable=global-statement
    if name == 'Default' and hasattr(model, 'scheduler'):
        config = {k: v for k, v in model.scheduler.config.items() if not k.startswith('_')}
        shared.log.debug(f'Sampler default {type(model.scheduler).__name__}: {config}')
        return model.scheduler
    config = find_sampler_config(name)
    if config is None:
        shared.log.error(f'Attempting to use unknown sampler: {name}')
        config = all_samplers[0]
    sampler = loaded_sampler
    if shared.backend == shared.Backend.ORIGINAL:
        if config != loaded_config:
            sampler = config.constructor(model)
            sampler.config = config
            sampler.name = name
            loaded_config = config
            shared.log.debug(f'Sampler: sampler="{name}" config={config.options}')
        sampler.initialize(p=None)
        loaded_sampler = sampler
    elif shared.backend == shared.Backend.DIFFUSERS:
        if config != loaded_config:
            sampler = config.constructor(model)
            loaded_config = config
            if not hasattr(model, 'scheduler_config'):
                model.scheduler_config = sampler.sampler.config.copy()
            shared.log.debug(f'Sampler: sampler="{sampler.name}" config={sampler.config}')
            loaded_sampler = sampler.sampler
        model.scheduler = loaded_sampler
    return loaded_sampler


def set_samplers():
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    samplers = visible_sampler_names()
    # samplers_for_img2img = [x for x in samplers if x.name != "PLMS"]
    samplers_for_img2img = samplers
    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name
