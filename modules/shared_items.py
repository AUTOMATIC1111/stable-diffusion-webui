import sys

from modules.shared_cmd_options import cmd_opts


def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    import modules.scripts

    return modules.scripts.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae

    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)


def refresh_vae_list():
    import modules.sd_vae

    modules.sd_vae.refresh_vae_list()


def cross_attention_optimizations():
    import modules.sd_hijack

    return ["Automatic"] + [x.title() for x in modules.sd_hijack.optimizers] + ["None"]


def sd_unet_items():
    import modules.sd_unet

    return ["Automatic"] + [x.label for x in modules.sd_unet.unet_options] + ["None"]


def refresh_unet_list():
    import modules.sd_unet

    modules.sd_unet.list_unets()


def list_checkpoint_tiles():
    import modules.sd_models
    return modules.sd_models.checkpoint_tiles()


def refresh_checkpoints():
    import modules.sd_models
    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers
    return modules.sd_samplers.all_samplers


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    from modules import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


ui_reorder_categories_builtin_items = [
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from modules import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"


class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        import modules.sd_models

        return modules.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models

        modules.sd_models.model_data.set_sd_model(value)


sys.modules['modules.shared'].__class__ = Shared
