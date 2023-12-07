import gradio as gr
from modules.ui_components import InputAccordion
import modules.scripts as scripts


class SoftInpaintingSettings:
    def __init__(self, mask_blend_power, mask_blend_scale, inpaint_detail_preservation):
        self.mask_blend_power = mask_blend_power
        self.mask_blend_scale = mask_blend_scale
        self.inpaint_detail_preservation = inpaint_detail_preservation

    def add_generation_params(self, dest):
        dest[enabled_gen_param_label] = True
        dest[gen_param_labels.mask_blend_power] = self.mask_blend_power
        dest[gen_param_labels.mask_blend_scale] = self.mask_blend_scale
        dest[gen_param_labels.inpaint_detail_preservation] = self.inpaint_detail_preservation


# ------------------- Methods -------------------


def latent_blend(soft_inpainting, a, b, t):
    """
    Interpolates two latent image representations according to the parameter t,
    where the interpolated vectors' magnitudes are also interpolated separately.
    The "detail_preservation" factor biases the magnitude interpolation towards
    the larger of the two magnitudes.
    """
    import torch

    # NOTE: We use inplace operations wherever possible.

    # [4][w][h] to [1][4][w][h]
    t2 = t.unsqueeze(0)
    # [4][w][h] to [1][1][w][h] - the [4] seem redundant.
    t3 = t[0].unsqueeze(0).unsqueeze(0)

    one_minus_t2 = 1 - t2
    one_minus_t3 = 1 - t3

    # Linearly interpolate the image vectors.
    a_scaled = a * one_minus_t2
    b_scaled = b * t2
    image_interp = a_scaled
    image_interp.add_(b_scaled)
    result_type = image_interp.dtype
    del a_scaled, b_scaled, t2, one_minus_t2

    # Calculate the magnitude of the interpolated vectors. (We will remove this magnitude.)
    # 64-bit operations are used here to allow large exponents.
    current_magnitude = torch.norm(image_interp, p=2, dim=1, keepdim=True).to(torch.float64).add_(0.00001)

    # Interpolate the powered magnitudes, then un-power them (bring them back to a power of 1).
    a_magnitude = torch.norm(a, p=2, dim=1, keepdim=True).to(torch.float64).pow_(
        soft_inpainting.inpaint_detail_preservation) * one_minus_t3
    b_magnitude = torch.norm(b, p=2, dim=1, keepdim=True).to(torch.float64).pow_(
        soft_inpainting.inpaint_detail_preservation) * t3
    desired_magnitude = a_magnitude
    desired_magnitude.add_(b_magnitude).pow_(1 / soft_inpainting.inpaint_detail_preservation)
    del a_magnitude, b_magnitude, t3, one_minus_t3

    # Change the linearly interpolated image vectors' magnitudes to the value we want.
    # This is the last 64-bit operation.
    image_interp_scaling_factor = desired_magnitude
    image_interp_scaling_factor.div_(current_magnitude)
    image_interp_scaling_factor = image_interp_scaling_factor.to(result_type)
    image_interp_scaled = image_interp
    image_interp_scaled.mul_(image_interp_scaling_factor)
    del current_magnitude
    del desired_magnitude
    del image_interp
    del image_interp_scaling_factor
    del result_type

    return image_interp_scaled


def get_modified_nmask(soft_inpainting, nmask, sigma):
    """
    Converts a negative mask representing the transparency of the original latent vectors being overlayed
    to a mask that is scaled according to the denoising strength for this step.

    Where:
        0 = fully opaque, infinite density, fully masked
        1 = fully transparent, zero density, fully unmasked

    We bring this transparency to a power, as this allows one to simulate N number of blending operations
    where N can be any positive real value. Using this one can control the balance of influence between
    the denoiser and the original latents according to the sigma value.

    NOTE: "mask" is not used
    """
    import torch
    return torch.pow(nmask, (sigma ** soft_inpainting.mask_blend_power) * soft_inpainting.mask_blend_scale)


def apply_adaptive_masks(
        latent_orig,
        latent_processed,
        overlay_images,
        width, height,
        paste_to):
    import torch
    import numpy as np
    import modules.processing as proc
    import modules.images as images
    from PIL import Image, ImageOps, ImageFilter

    # TODO: Bias the blending according to the latent mask, add adjustable parameter for bias control.
    # latent_mask = p.nmask[0].float().cpu()
    # convert the original mask into a form we use to scale distances for thresholding
    # mask_scalar = 1-(torch.clamp(latent_mask, min=0, max=1) ** (p.mask_blend_scale / 2))
    # mask_scalar = mask_scalar / (1.00001-mask_scalar)
    # mask_scalar = mask_scalar.numpy()

    latent_distance = torch.norm(latent_processed - latent_orig, p=2, dim=1)

    kernel, kernel_center = images.get_gaussian_kernel(stddev_radius=1.5, max_radius=2)

    masks_for_overlay = []

    for i, (distance_map, overlay_image) in enumerate(zip(latent_distance, overlay_images)):
        converted_mask = distance_map.float().cpu().numpy()
        converted_mask = images.weighted_histogram_filter(converted_mask, kernel, kernel_center,
                                                          percentile_min=0.9, percentile_max=1, min_width=1)
        converted_mask = images.weighted_histogram_filter(converted_mask, kernel, kernel_center,
                                                          percentile_min=0.25, percentile_max=0.75, min_width=1)

        # The distance at which opacity of original decreases to 50%
        # half_weighted_distance = 1  # * mask_scalar
        # converted_mask = converted_mask / half_weighted_distance

        converted_mask = 1 / (1 + converted_mask ** 2)
        converted_mask = images.smootherstep(converted_mask)
        converted_mask = 1 - converted_mask
        converted_mask = 255. * converted_mask
        converted_mask = converted_mask.astype(np.uint8)
        converted_mask = Image.fromarray(converted_mask)
        converted_mask = images.resize_image(2, converted_mask, width, height)
        converted_mask = proc.create_binary_mask(converted_mask, round=False)

        # Remove aliasing artifacts using a gaussian blur.
        converted_mask = converted_mask.filter(ImageFilter.GaussianBlur(radius=4))

        # Expand the mask to fit the whole image if needed.
        if paste_to is not None:
            converted_mask = proc.uncrop(converted_mask,
                                         (overlay_image.width, overlay_image.height),
                                         paste_to)

        masks_for_overlay.append(converted_mask)

        image_masked = Image.new('RGBa', (overlay_image.width, overlay_image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"),
                           mask=ImageOps.invert(converted_mask.convert('L')))

        overlay_images[i] = image_masked.convert('RGBA')

    return masks_for_overlay


def apply_masks(
        soft_inpainting,
        nmask,
        overlay_images,
        width, height,
        paste_to):
    import torch
    import numpy as np
    import modules.processing as proc
    import modules.images as images
    from PIL import Image, ImageOps, ImageFilter

    converted_mask = nmask[0].float()
    converted_mask = torch.clamp(converted_mask, min=0, max=1).pow_(soft_inpainting.mask_blend_scale / 2)
    converted_mask = 255. * converted_mask
    converted_mask = converted_mask.cpu().numpy().astype(np.uint8)
    converted_mask = Image.fromarray(converted_mask)
    converted_mask = images.resize_image(2, converted_mask, width, height)
    converted_mask = proc.create_binary_mask(converted_mask, round=False)

    # Remove aliasing artifacts using a gaussian blur.
    converted_mask = converted_mask.filter(ImageFilter.GaussianBlur(radius=4))

    # Expand the mask to fit the whole image if needed.
    if paste_to is not None:
        converted_mask = proc.uncrop(converted_mask,
                                     (width, height),
                                     paste_to)

    masks_for_overlay = []

    for i, overlay_image in enumerate(overlay_images):
        masks_for_overlay[i] = converted_mask

        image_masked = Image.new('RGBa', (overlay_image.width, overlay_image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"),
                           mask=ImageOps.invert(converted_mask.convert('L')))

        overlay_images[i] = image_masked.convert('RGBA')

    return masks_for_overlay


# ------------------- Constants -------------------


default = SoftInpaintingSettings(1, 0.5, 4)

enabled_ui_label = "Soft inpainting"
enabled_gen_param_label = "Soft inpainting enabled"
enabled_el_id = "soft_inpainting_enabled"

ui_labels = SoftInpaintingSettings(
    "Schedule bias",
    "Preservation strength",
    "Transition contrast boost")

ui_info = SoftInpaintingSettings(
    "Shifts when preservation of original content occurs during denoising.",
    "How strongly partially masked content should be preserved.",
    "Amplifies the contrast that may be lost in partially masked regions.")

gen_param_labels = SoftInpaintingSettings(
    "Soft inpainting schedule bias",
    "Soft inpainting preservation strength",
    "Soft inpainting transition contrast boost")

el_ids = SoftInpaintingSettings(
    "mask_blend_power",
    "mask_blend_scale",
    "inpaint_detail_preservation")


class Script(scripts.Script):

    def __init__(self):
        self.masks_for_overlay = None
        self.overlay_images = None

    def title(self):
        return "Soft Inpainting"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        if not is_img2img:
            return

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

        self.infotext_fields = [(soft_inpainting_enabled, enabled_gen_param_label),
                                (result.mask_blend_power, gen_param_labels.mask_blend_power),
                                (result.mask_blend_scale, gen_param_labels.mask_blend_scale),
                                (result.inpaint_detail_preservation, gen_param_labels.inpaint_detail_preservation)]

        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)

        return [soft_inpainting_enabled,
                result.mask_blend_power,
                result.mask_blend_scale,
                result.inpaint_detail_preservation]

    def process(self, p, enabled, power, scale, detail_preservation):
        if not enabled:
            return

        # Shut off the rounding it normally does.
        p.mask_round = False

        settings = SoftInpaintingSettings(power, scale, detail_preservation)

        # p.extra_generation_params["Mask rounding"] = False
        settings.add_generation_params(p.extra_generation_params)

    def on_mask_blend(self, p, mba: scripts.MaskBlendArgs, enabled, power, scale, detail_preservation):
        if not enabled:
            return

        if mba.sigma is None:
            mba.blended_latent = mba.current_latent
            return

        settings = SoftInpaintingSettings(power, scale, detail_preservation)

        # todo: Why is sigma 2D? Both values are the same.
        mba.blended_latent = latent_blend(settings,
                                          mba.init_latent,
                                          mba.current_latent,
                                          get_modified_nmask(settings, mba.nmask, mba.sigma[0]))

    def post_sample(self, p, ps: scripts.PostSampleArgs, enabled, power, scale, detail_preservation):
        if not enabled:
            return

        settings = SoftInpaintingSettings(power, scale, detail_preservation)

        from modules import images
        from modules.shared import opts

        # since the original code puts holes in the existing overlay images,
        # we have to rebuild them.
        self.overlay_images = []
        for img in p.init_images:

            image = images.flatten(img, opts.img2img_background_color)

            if p.paste_to is None and p.resize_mode != 3:
                image = images.resize_image(p.resize_mode, image, p.width, p.height)

            self.overlay_images.append(image.convert('RGBA'))

        if getattr(ps.samples, 'already_decoded', False):
            self.masks_for_overlay = apply_masks(soft_inpainting=settings,
                                                 nmask=p.nmask,
                                                 overlay_images=self.overlay_images,
                                                 width=p.width,
                                                 height=p.height,
                                                 paste_to=p.paste_to)
        else:
            self.masks_for_overlay = apply_adaptive_masks(latent_orig=p.init_latent,
                                                          latent_processed=ps.samples,
                                                          overlay_images=self.overlay_images,
                                                          width=p.width,
                                                          height=p.height,
                                                          paste_to=p.paste_to)


    def postprocess_maskoverlay(self, p, ppmo: scripts.PostProcessMaskOverlayArgs, enabled, power, scale, detail_preservation):
        if not enabled:
            return

        ppmo.mask_for_overlay = self.masks_for_overlay[ppmo.index]
        ppmo.overlay_image = self.overlay_images[ppmo.index]
