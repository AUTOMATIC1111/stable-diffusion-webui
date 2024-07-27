from __future__ import annotations

import functools
import logging
from modules import sd_samplers_kdiffusion, sd_samplers_timesteps, sd_samplers_lcm, shared, sd_samplers_common, sd_schedulers

# imports for functions that previously were here and are used by other modules
samples_to_image_grid = sd_samplers_common.samples_to_image_grid
sample_to_image = sd_samplers_common.sample_to_image

all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
    *sd_samplers_lcm.samplers_data_lcm,
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers: list[sd_samplers_common.SamplerData] = []
samplers_for_img2img: list[sd_samplers_common.SamplerData] = []
samplers_map = {}
samplers_hidden = {}


def find_sampler_config(name):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    return config


def create_sampler(name, model):
    config = find_sampler_config(name)

    assert config is not None, f'bad sampler name: {name}'

    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise Exception(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers():
    global samplers, samplers_for_img2img, samplers_hidden

    samplers_hidden = set(shared.opts.hide_samplers)
    samplers = all_samplers
    samplers_for_img2img = all_samplers

    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name


def visible_sampler_names():
    return [x.name for x in samplers if x.name not in samplers_hidden]


def visible_samplers():
    return [x for x in samplers if x.name not in samplers_hidden]


def get_sampler_from_infotext(d: dict):
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[0]


def get_scheduler_from_infotext(d: dict):
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[1]


def get_hr_sampler_and_scheduler(d: dict):
    hr_sampler = d.get("Hires sampler", "Use same sampler")
    sampler = d.get("Sampler") if hr_sampler == "Use same sampler" else hr_sampler

    hr_scheduler = d.get("Hires schedule type", "Use same scheduler")
    scheduler = d.get("Schedule type") if hr_scheduler == "Use same scheduler" else hr_scheduler

    sampler, scheduler = get_sampler_and_scheduler(sampler, scheduler)

    sampler = sampler if sampler != d.get("Sampler") else "Use same sampler"
    scheduler = scheduler if scheduler != d.get("Schedule type") else "Use same scheduler"

    return sampler, scheduler


def get_hr_sampler_from_infotext(d: dict):
    return get_hr_sampler_and_scheduler(d)[0]


def get_hr_scheduler_from_infotext(d: dict):
    return get_hr_sampler_and_scheduler(d)[1]


@functools.cache
def get_sampler_and_scheduler(sampler_name, scheduler_name, *, convert_automatic=True):
    default_sampler = samplers[0]
    found_scheduler = sd_schedulers.schedulers_map.get(scheduler_name, sd_schedulers.schedulers[0])

    name = sampler_name or default_sampler.name

    for scheduler in sd_schedulers.schedulers:
        name_options = [scheduler.label, scheduler.name, *(scheduler.aliases or [])]

        for name_option in name_options:
            if name.endswith(" " + name_option):
                found_scheduler = scheduler
                name = name[0:-(len(name_option) + 1)]
                break

    sampler = all_samplers_map.get(name, default_sampler)

    # revert back to Automatic if it's the default scheduler for the selected sampler
    if convert_automatic and sampler.options.get('scheduler', None) == found_scheduler.name:
        found_scheduler = sd_schedulers.schedulers[0]

    return sampler.name, found_scheduler.label


def fix_p_invalid_sampler_and_scheduler(p):
    i_sampler_name, i_scheduler = p.sampler_name, p.scheduler
    p.sampler_name, p.scheduler = get_sampler_and_scheduler(p.sampler_name, p.scheduler, convert_automatic=False)
    if p.sampler_name != i_sampler_name or i_scheduler != p.scheduler:
        logging.warning(f'Sampler Scheduler autocorrection: "{i_sampler_name}" -> "{p.sampler_name}", "{i_scheduler}" -> "{p.scheduler}"')


set_samplers()
