class SoftInpaintingSettings:
    def __init__(self, mask_blend_power, mask_blend_scale, inpaint_detail_preservation):
        self.mask_blend_power = mask_blend_power
        self.mask_blend_scale = mask_blend_scale
        self.inpaint_detail_preservation = inpaint_detail_preservation

    def get_paste_fields(self):
        return [
            (self.mask_blend_power, gen_param_labels.mask_blend_power),
            (self.mask_blend_scale, gen_param_labels.mask_blend_scale),
            (self.inpaint_detail_preservation, gen_param_labels.inpaint_detail_preservation),
        ]

    def add_generation_params(self, dest):
        dest[enabled_gen_param_label] = True
        dest[gen_param_labels.mask_blend_power] = self.mask_blend_power
        dest[gen_param_labels.mask_blend_scale] = self.mask_blend_scale
        dest[gen_param_labels.inpaint_detail_preservation] = self.inpaint_detail_preservation


enabled_ui_label = "Soft inpainting"
enabled_gen_param_label = "Soft inpainting enabled"
enabled_el_id = "soft_inpainting_enabled"

default = SoftInpaintingSettings(1, 0.5, 4)
ui_labels = SoftInpaintingSettings("Schedule bias", "Preservation strength", "Transition contrast boost")

ui_info = SoftInpaintingSettings(
    mask_blend_power="Shifts when preservation of original content occurs during denoising.",
                     # "Below 1: Stronger preservation near the end (with low sigma)\n"
                     # "1: Balanced (proportional to sigma)\n"
                     # "Above 1: Stronger preservation in the beginning (with high sigma)",
    mask_blend_scale="How strongly partially masked content should be preserved.",
                     # "Low values: Favors generated content.\n"
                     # "High values: Favors original content.",
    inpaint_detail_preservation="Amplifies the contrast that may be lost in partially masked regions.")

gen_param_labels = SoftInpaintingSettings("Soft inpainting schedule bias", "Soft inpainting preservation strength", "Soft inpainting transition contrast boost")
el_ids = SoftInpaintingSettings("mask_blend_power", "mask_blend_scale", "inpaint_detail_preservation")


def gradio_ui():
    import gradio as gr
    from modules.ui_components import InputAccordion
    """
            with InputAccordion(False, label="Refiner", elem_id=self.elem_id("enable")) as enable_refiner:
            with gr.Row():
                refiner_checkpoint = gr.Dropdown(label='Checkpoint', elem_id=self.elem_id("checkpoint"), choices=sd_models.checkpoint_tiles(), value='', tooltip="switch to another model in the middle of generation")
                create_refresh_button(refiner_checkpoint, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, self.elem_id("checkpoint_refresh"))

                refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01, elem_id=self.elem_id("switch_at"), tooltip="fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation")

    """
    with InputAccordion(False, label=enabled_ui_label, elem_id=enabled_el_id) as soft_inpainting_enabled:
        with gr.Group():
            gr.Markdown(
                """
                Soft inpainting allows you to **seamlessly blend original content with inpainted content** according to the mask opacity.
                **High _Mask blur_** values are recommended!
                """)

            result = SoftInpaintingSettings(
                gr.Slider(label=ui_labels.mask_blend_power,
                          info=ui_info.mask_blend_power,
                          minimum=0,
                          maximum=8,
                          step=0.1,
                          value=default.mask_blend_power,
                          elem_id=el_ids.mask_blend_power),
                gr.Slider(label=ui_labels.mask_blend_scale,
                          info=ui_info.mask_blend_scale,
                          minimum=0,
                          maximum=8,
                          step=0.05,
                          value=default.mask_blend_scale,
                          elem_id=el_ids.mask_blend_scale),
                gr.Slider(label=ui_labels.inpaint_detail_preservation,
                          info=ui_info.inpaint_detail_preservation,
                          minimum=1,
                          maximum=32,
                          step=0.5,
                          value=default.inpaint_detail_preservation,
                          elem_id=el_ids.inpaint_detail_preservation))

            with gr.Accordion("Help", open=False):
                gr.Markdown(
                    f"""
                    ### {ui_labels.mask_blend_power}
                    
                    The blending strength of original content is scaled proportionally with the decreasing noise level values at each step (sigmas).
                    This ensures that the influence of the denoiser and original content preservation is roughly balanced at each step.
                    This balance can be shifted using this parameter, controlling whether earlier or later steps have stronger preservation.
                    
                    - **Below 1**: Stronger preservation near the end (with low sigma)
                    - **1**: Balanced (proportional to sigma)
                    - **Above 1**: Stronger preservation in the beginning (with high sigma)
                    """)
                gr.Markdown(
                    f"""
                    ### {ui_labels.mask_blend_scale}
                    
                    Skews whether partially masked image regions should be more likely to preserve the original content or favor inpainted content.
                    This may need to be adjusted depending on the {ui_labels.mask_blend_power}, CFG Scale, prompt and Denoising strength.
                    
                    - **Low values**: Favors generated content.
                    - **High values**: Favors original content.
                    """)
                gr.Markdown(
                    f"""
                    ### {ui_labels.inpaint_detail_preservation}
                    
                    This parameter controls how the original latent vectors and denoised latent vectors are interpolated.
                    With higher values, the magnitude of the resulting blended vector will be closer to the maximum of the two interpolated vectors.
                    This can prevent the loss of contrast that occurs with linear interpolation.
                    
                    - **Low values**: Softer blending, details may fade.
                    - **High values**: Stronger contrast, may over-saturate colors.
                    """)

    return (
        [
            soft_inpainting_enabled,
            result.mask_blend_power,
            result.mask_blend_scale,
            result.inpaint_detail_preservation
        ],
        [
            (soft_inpainting_enabled, enabled_gen_param_label),
            (result.mask_blend_power, gen_param_labels.mask_blend_power),
            (result.mask_blend_scale, gen_param_labels.mask_blend_scale),
            (result.inpaint_detail_preservation, gen_param_labels.inpaint_detail_preservation)
        ]
    )
