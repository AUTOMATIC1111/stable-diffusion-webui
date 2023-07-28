import math

import numpy as np
import skimage

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import images
from modules.processing import Processed, process_images
from modules.shared import opts, state


# this function is taken from https://github.com/parlance-zz/g-diffuser-bot
def get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    # helper fft routines that keep ortho normalization and auto-shift before and after fft
    def _fft2(data):
        if data.ndim > 2:  # has channels
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:  # one channel
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    def _ifft2(data):
        if data.ndim > 2:  # has channels
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:  # one channel
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft

    def _get_gaussian_window(width, height, std=3.14, mode=0):
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)  # hey wait a minute that's not gaussian

        return window

    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # create a generator with a static seed to make outpainting deterministic / only follow global seed
    rng = np.random.default_rng(0)

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = rng.random((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    matched_noise = shaped_noise[:]

    return np.clip(matched_noise, 0., 1.)



class Script(scripts.Script):
    def title(self):
        return "Outpainting mk2"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>")

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128, elem_id=self.elem_id("pixels"))
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8, elem_id=self.elem_id("mask_blur"))
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'], elem_id=self.elem_id("direction"))
        noise_q = gr.Slider(label="Fall-off exponent (lower=higher detail)", minimum=0.0, maximum=4.0, step=0.01, value=1.0, elem_id=self.elem_id("noise_q"))
        color_variation = gr.Slider(label="Color variation", minimum=0.0, maximum=1.0, step=0.01, value=0.05, elem_id=self.elem_id("color_variation"))

        return [info, pixels, mask_blur, direction, noise_q, color_variation]

    def run(self, p, _, pixels, mask_blur, direction, noise_q, color_variation):
        initial_seed_and_info = [None, None]

        process_width = p.width
        process_height = p.height

        p.inpaint_full_res = False
        p.inpainting_fill = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

        if left > 0 or right > 0:
            mask_blur_x = mask_blur
        else:
            mask_blur_x = 0

        if up > 0 or down > 0:
            mask_blur_y = mask_blur
        else:
            mask_blur_y = 0

        p.mask_blur_x = mask_blur_x*4
        p.mask_blur_y = mask_blur_y*4

        init_img = p.init_images[0]
        target_w = math.ceil((init_img.width + left + right) / 64) * 64
        target_h = math.ceil((init_img.height + up + down) / 64) * 64

        if left > 0:
            left = left * (target_w - init_img.width) // (left + right)

        if right > 0:
            right = target_w - init_img.width - left

        if up > 0:
            up = up * (target_h - init_img.height) // (up + down)

        if down > 0:
            down = target_h - init_img.height - up

        def expand(init, count, expand_pixels, is_left=False, is_right=False, is_top=False, is_bottom=False):
            is_horiz = is_left or is_right
            is_vert = is_top or is_bottom
            pixels_horiz = expand_pixels if is_horiz else 0
            pixels_vert = expand_pixels if is_vert else 0

            images_to_process = []
            output_images = []
            for n in range(count):
                res_w = init[n].width + pixels_horiz
                res_h = init[n].height + pixels_vert
                process_res_w = math.ceil(res_w / 64) * 64
                process_res_h = math.ceil(res_h / 64) * 64

                img = Image.new("RGB", (process_res_w, process_res_h))
                img.paste(init[n], (pixels_horiz if is_left else 0, pixels_vert if is_top else 0))
                mask = Image.new("RGB", (process_res_w, process_res_h), "white")
                draw = ImageDraw.Draw(mask)
                draw.rectangle((
                    expand_pixels + mask_blur_x if is_left else 0,
                    expand_pixels + mask_blur_y if is_top else 0,
                    mask.width - expand_pixels - mask_blur_x if is_right else res_w,
                    mask.height - expand_pixels - mask_blur_y if is_bottom else res_h,
                ), fill="black")

                np_image = (np.asarray(img) / 255.0).astype(np.float64)
                np_mask = (np.asarray(mask) / 255.0).astype(np.float64)
                noised = get_matched_noise(np_image, np_mask, noise_q, color_variation)
                output_images.append(Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB"))

                target_width = min(process_width, init[n].width + pixels_horiz) if is_horiz else img.width
                target_height = min(process_height, init[n].height + pixels_vert) if is_vert else img.height
                p.width = target_width if is_horiz else img.width
                p.height = target_height if is_vert else img.height

                crop_region = (
                    0 if is_left else output_images[n].width - target_width,
                    0 if is_top else output_images[n].height - target_height,
                    target_width if is_left else output_images[n].width,
                    target_height if is_top else output_images[n].height,
                )
                mask = mask.crop(crop_region)
                p.image_mask = mask

                image_to_process = output_images[n].crop(crop_region)
                images_to_process.append(image_to_process)

            p.init_images = images_to_process

            latent_mask = Image.new("RGB", (p.width, p.height), "white")
            draw = ImageDraw.Draw(latent_mask)
            draw.rectangle((
                expand_pixels + mask_blur_x * 2 if is_left else 0,
                expand_pixels + mask_blur_y * 2 if is_top else 0,
                mask.width - expand_pixels - mask_blur_x * 2 if is_right else res_w,
                mask.height - expand_pixels - mask_blur_y * 2 if is_bottom else res_h,
            ), fill="black")
            p.latent_mask = latent_mask

            proc = process_images(p)

            if initial_seed_and_info[0] is None:
                initial_seed_and_info[0] = proc.seed
                initial_seed_and_info[1] = proc.info

            for n in range(count):
                output_images[n].paste(proc.images[n], (0 if is_left else output_images[n].width - proc.images[n].width, 0 if is_top else output_images[n].height - proc.images[n].height))
                output_images[n] = output_images[n].crop((0, 0, res_w, res_h))

            return output_images

        batch_count = p.n_iter
        batch_size = p.batch_size
        p.n_iter = 1
        state.job_count = batch_count * ((1 if left > 0 else 0) + (1 if right > 0 else 0) + (1 if up > 0 else 0) + (1 if down > 0 else 0))
        all_processed_images = []

        for i in range(batch_count):
            imgs = [init_img] * batch_size
            state.job = f"Batch {i + 1} out of {batch_count}"

            if left > 0:
                imgs = expand(imgs, batch_size, left, is_left=True)
            if right > 0:
                imgs = expand(imgs, batch_size, right, is_right=True)
            if up > 0:
                imgs = expand(imgs, batch_size, up, is_top=True)
            if down > 0:
                imgs = expand(imgs, batch_size, down, is_bottom=True)

            all_processed_images += imgs

        all_images = all_processed_images

        combined_grid_image = images.image_grid(all_processed_images)
        unwanted_grid_because_of_img_count = len(all_processed_images) < 2 and opts.grid_only_if_multiple
        if opts.return_grid and not unwanted_grid_because_of_img_count:
            all_images = [combined_grid_image] + all_processed_images

        res = Processed(p, all_images, initial_seed_and_info[0], initial_seed_and_info[1])

        if opts.samples_save:
            for img in all_processed_images:
                images.save_image(img, p.outpath_samples, "", res.seed, p.prompt, opts.samples_format, info=res.info, p=p)

        if opts.grid_save and not unwanted_grid_because_of_img_count:
            images.save_image(combined_grid_image, p.outpath_grids, "grid", res.seed, p.prompt, opts.grid_format, info=res.info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

        return res
