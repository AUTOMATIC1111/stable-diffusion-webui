import math

import numpy as np
import skimage

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import images, processing, devices
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state

#  https://github.com/parlance-zz/g-diffuser-bot
def expand(x, dir, amount, power=0.75):
    is_left = dir == 3
    is_right = dir == 1
    is_up = dir == 0
    is_down = dir == 2

    if is_left or is_right:
        noise = np.zeros((x.shape[0], amount, 3), dtype=float)
        indexes = np.random.random((x.shape[0], amount)) ** power * (1 - np.arange(amount) / amount)
        if is_right:
            indexes = 1 - indexes
        indexes = (indexes * (x.shape[1] - 1)).astype(int)

        for row in range(x.shape[0]):
            if is_left:
                noise[row] = x[row][indexes[row]]
            else:
                noise[row] = np.flip(x[row][indexes[row]], axis=0)

        x = np.concatenate([noise, x] if is_left else [x, noise], axis=1)
        return x

    if is_up or is_down:
        noise = np.zeros((amount, x.shape[1], 3), dtype=float)
        indexes = np.random.random((x.shape[1], amount)) ** power * (1 - np.arange(amount) / amount)
        if is_down:
            indexes = 1 - indexes
        indexes = (indexes * x.shape[0] - 1).astype(int)

        for row in range(x.shape[1]):
            if is_up:
                noise[:, row] = x[:, row][indexes[row]]
            else:
                noise[:, row] = np.flip(x[:, row][indexes[row]], axis=0)

        x = np.concatenate([noise, x] if is_up else [x, noise], axis=0)
        return x


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

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
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

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128)
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8, visible=False)
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'])
        noise_q = gr.Slider(label="Fall-off exponent (lower=higher detail)", minimum=0.0, maximum=4.0, step=0.01, value=1.0)
        color_variation = gr.Slider(label="Color variation", minimum=0.0, maximum=1.0, step=0.01, value=0.05)

        return [info, pixels, mask_blur, direction, noise_q, color_variation]

    def run(self, p, _, pixels, mask_blur, direction, noise_q, color_variation):
        initial_seed_and_info = [None, None]

        process_width = p.width
        process_height = p.height

        p.mask_blur = mask_blur*4
        p.inpaint_full_res = False
        p.inpainting_fill = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

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

        init_image = p.init_images[0]

        state.job_count = (1 if left > 0 else 0) + (1 if right > 0 else 0) + (1 if up > 0 else 0) + (1 if down > 0 else 0)

        def expand(init, expand_pixels, is_left=False, is_right=False, is_top=False, is_bottom=False):
            is_horiz = is_left or is_right
            is_vert = is_top or is_bottom
            pixels_horiz = expand_pixels if is_horiz else 0
            pixels_vert = expand_pixels if is_vert else 0

            res_w = init.width + pixels_horiz
            res_h = init.height + pixels_vert
            process_res_w = math.ceil(res_w / 64) * 64
            process_res_h = math.ceil(res_h / 64) * 64

            img = Image.new("RGB", (process_res_w, process_res_h))
            img.paste(init, (pixels_horiz if is_left else 0, pixels_vert if is_top else 0))
            mask = Image.new("RGB", (process_res_w, process_res_h), "white")
            draw = ImageDraw.Draw(mask)
            draw.rectangle((
                expand_pixels + mask_blur if is_left else 0,
                expand_pixels + mask_blur if is_top else 0,
                mask.width - expand_pixels - mask_blur if is_right else res_w,
                mask.height - expand_pixels - mask_blur if is_bottom else res_h,
            ), fill="black")

            np_image = (np.asarray(img) / 255.0).astype(np.float64)
            np_mask = (np.asarray(mask) / 255.0).astype(np.float64)
            noised = get_matched_noise(np_image, np_mask, noise_q, color_variation)
            out = Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")

            target_width = min(process_width, init.width + pixels_horiz) if is_horiz else img.width
            target_height = min(process_height, init.height + pixels_vert) if is_vert else img.height

            crop_region = (
                0 if is_left else out.width - target_width,
                0 if is_top else out.height - target_height,
                target_width if is_left else out.width,
                target_height if is_top else out.height,
            )

            image_to_process = out.crop(crop_region)
            mask = mask.crop(crop_region)

            p.width = target_width if is_horiz else img.width
            p.height = target_height if is_vert else img.height
            p.init_images = [image_to_process]
            p.image_mask = mask

            latent_mask = Image.new("RGB", (p.width, p.height), "white")
            draw = ImageDraw.Draw(latent_mask)
            draw.rectangle((
                expand_pixels + mask_blur * 2 if is_left else 0,
                expand_pixels + mask_blur * 2 if is_top else 0,
                mask.width - expand_pixels - mask_blur * 2 if is_right else res_w,
                mask.height - expand_pixels - mask_blur * 2 if is_bottom else res_h,
            ), fill="black")
            p.latent_mask = latent_mask

            proc = process_images(p)
            proc_img = proc.images[0]

            if initial_seed_and_info[0] is None:
                initial_seed_and_info[0] = proc.seed
                initial_seed_and_info[1] = proc.info

            out.paste(proc_img, (0 if is_left else out.width - proc_img.width, 0 if is_top else out.height - proc_img.height))
            out = out.crop((0, 0, res_w, res_h))
            return out

        img = init_image

        if left > 0:
            img = expand(img, left, is_left=True)
        if right > 0:
            img = expand(img, right, is_right=True)
        if up > 0:
            img = expand(img, up, is_top=True)
        if down > 0:
            img = expand(img, down, is_bottom=True)

        res = Processed(p, [img], initial_seed_and_info[0], initial_seed_and_info[1])

        if opts.samples_save:
            images.save_image(img, p.outpath_samples, "", res.seed, p.prompt, opts.grid_format, info=res.info, p=p)

        return res

