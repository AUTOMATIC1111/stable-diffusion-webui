import html
import sys

from modules import script_callbacks, scripts, ui_components
from modules.options import OptionHTML, OptionInfo
from modules.shared_cmd_options import cmd_opts


def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


def dat_models_names():
    import modules.dat_model
    return [x.name for x in modules.dat_model.get_dat_models(None)]


def postprocessing_scripts(filter_out_extra_only=False, filter_out_main_ui_only=False):
    import modules.scripts
    return list(filter(
        lambda s: (not filter_out_extra_only or not s.extra_only) and (not filter_out_main_ui_only or not s.main_ui_only),
        modules.scripts.scripts_postproc.scripts,
    ))


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


def list_checkpoint_tiles(use_short=False):
    import modules.sd_models
    return modules.sd_models.checkpoint_tiles(use_short)


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


def get_infotext_names():
    from modules import infotext_utils, shared
    res = {}

    for info in shared.opts.data_labels.values():
        if info.infotext:
            res[info.infotext] = 1

    for tab_data in infotext_utils.paste_fields.values():
        for _, name in tab_data.get("fields") or []:
            if isinstance(name, str):
                res[name] = 1

    return list(res)


ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
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


def callbacks_order_settings():
    options = {
        "sd_vae_explanation": OptionHTML("""
    For categories below, callbacks added to dropdowns happen before others, in order listed.
    """),

    }

    callback_options = {}

    for category, _ in script_callbacks.enumerate_callbacks():
        callback_options[category] = script_callbacks.ordered_callbacks(category, enable_user_sort=False)

    for method_name in scripts.scripts_txt2img.callback_names:
        callback_options["script_" + method_name] = scripts.scripts_txt2img.create_ordered_callbacks_list(method_name, enable_user_sort=False)

    for method_name in scripts.scripts_img2img.callback_names:
        callbacks = callback_options.get("script_" + method_name, [])

        for addition in scripts.scripts_img2img.create_ordered_callbacks_list(method_name, enable_user_sort=False):
            if any(x.name == addition.name for x in callbacks):
                continue

            callbacks.append(addition)

        callback_options["script_" + method_name] = callbacks

    for category, callbacks in callback_options.items():
        if not callbacks:
            continue

        option_info = OptionInfo([], f"{category} callback priority", ui_components.DropdownMulti, {"choices": [x.name for x in callbacks]})
        option_info.needs_restart()
        option_info.html("<div class='info'>Default order: <ol>" + "".join(f"<li>{html.escape(x.name)}</li>\n" for x in callbacks) + "</ol></div>")
        options['prioritized_callbacks_' + category] = option_info

    return options


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
