import hypertile
from modules import scripts, script_callbacks, shared
from scripts.hypertile_xyz import add_axis_options


class ScriptHypertile(scripts.Script):
    name = "Hypertile"

    def title(self):
        return self.name

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        hypertile.set_hypertile_seed(p.all_seeds[0])

        configure_hypertile(p.width, p.height, enable_unet=shared.opts.hypertile_enable_unet)

    def before_hr(self, p, *args):
        # exclusive hypertile seed for the second pass
        if not shared.opts.hypertile_enable_unet:
            hypertile.set_hypertile_seed(p.all_seeds[0])
        configure_hypertile(p.hr_upscale_to_x, p.hr_upscale_to_y, enable_unet=shared.opts.hypertile_enable_unet_secondpass)


def configure_hypertile(width, height, enable_unet=True):
    hypertile.hypertile_hook_model(
        shared.sd_model.first_stage_model,
        width,
        height,
        swap_size=shared.opts.hypertile_swap_size_vae,
        max_depth=shared.opts.hypertile_max_depth_vae,
        tile_size_max=shared.opts.hypertile_max_tile_vae,
        enable=shared.opts.hypertile_enable_vae,
    )

    hypertile.hypertile_hook_model(
        shared.sd_model.model,
        width,
        height,
        swap_size=shared.opts.hypertile_swap_size_unet,
        max_depth=shared.opts.hypertile_max_depth_unet,
        tile_size_max=shared.opts.hypertile_max_tile_unet,
        enable=enable_unet,
        is_sdxl=shared.sd_model.is_sdxl
    )


def on_ui_settings():
    import gradio as gr

    options = {
        "hypertile_explanation": shared.OptionHTML("""
    <a href='https://github.com/tfernd/HyperTile'>Hypertile</a> optimizes the self-attention layer within U-Net and VAE models,
    resulting in a reduction in computation time ranging from 1 to 4 times. The larger the generated image is, the greater the
    benefit.
    """),

        "hypertile_enable_unet": shared.OptionInfo(False, "Enable Hypertile U-Net").info("noticeable change in details of the generated picture; if enabled, overrides the setting below"),
        "hypertile_enable_unet_secondpass": shared.OptionInfo(False, "Enable Hypertile U-Net for hires fix second pass"),
        "hypertile_max_depth_unet": shared.OptionInfo(3, "Hypertile U-Net max depth", gr.Slider, {"minimum": 0, "maximum": 3, "step": 1}),
        "hypertile_max_tile_unet": shared.OptionInfo(256, "Hypertile U-net max tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
        "hypertile_swap_size_unet": shared.OptionInfo(3, "Hypertile U-net swap size", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),

        "hypertile_enable_vae": shared.OptionInfo(False, "Enable Hypertile VAE").info("minimal change in the generated picture"),
        "hypertile_max_depth_vae": shared.OptionInfo(3, "Hypertile VAE max depth", gr.Slider, {"minimum": 0, "maximum": 3, "step": 1}),
        "hypertile_max_tile_vae": shared.OptionInfo(128, "Hypertile VAE max tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
        "hypertile_swap_size_vae": shared.OptionInfo(3, "Hypertile VAE swap size ", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),
    }

    for name, opt in options.items():
        opt.section = ('hypertile', "Hypertile")
        shared.opts.add_option(name, opt)


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_ui(add_axis_options)
