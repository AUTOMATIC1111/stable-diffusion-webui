import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import images, processing
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state



class Script(scripts.Script):
    def title(self):
        return "Poor man's outpainting"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=128, step=8)
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)
        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", visible=False)
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'])

        return [pixels, mask_blur, inpainting_fill, direction]

    def run(self, p, pixels, mask_blur, inpainting_fill, direction):
        initial_seed = None
        initial_info = None

        p.mask_blur = mask_blur
        p.inpainting_fill = inpainting_fill
        p.inpaint_full_res = False

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

        init_img = p.init_images[0]
        target_w = math.ceil((init_img.width + left + right) / 64) * 64
        target_h = math.ceil((init_img.height + up + down) / 64) * 64

        if left > 0:
            left = left * (target_w - init_img.width) // (left + right)
        right = target_w - init_img.width - left

        if up > 0:
            up = up * (target_h - init_img.height) // (up + down)
        down = target_h - init_img.height - up

        img = Image.new("RGB", (target_w, target_h))
        img.paste(init_img, (left, up))

        mask = Image.new("L", (img.width, img.height), "white")
        draw = ImageDraw.Draw(mask)
        draw.rectangle((
            left + (mask_blur * 2 if left > 0 else 0),
            up + (mask_blur * 2 if up > 0 else 0),
            mask.width - right - (mask_blur * 2 if right > 0 else 0),
            mask.height - down - (mask_blur * 2 if down > 0 else 0)
        ), fill="black")

        latent_mask = Image.new("L", (img.width, img.height), "white")
        latent_draw = ImageDraw.Draw(latent_mask)
        latent_draw.rectangle((left + left//2, up + up//2, mask.width - right - right//2, mask.height - down - down//2), fill="black")

        processing.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_latent_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=pixels)

        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_mask = []
        work_latent_mask = []
        work_results = []

        for (_, _, row), (_, _, row_mask), (_, _, row_latent_mask) in zip(grid.tiles, grid_mask.tiles, grid_latent_mask.tiles):
            for tiledata, tiledata_mask, tiledata_latent_mask in zip(row, row_mask, row_latent_mask):
                work.append(tiledata[2])
                work_mask.append(tiledata_mask[2])
                work_latent_mask.append(tiledata_latent_mask[2])

        batch_count = len(work)
        print(f"Poor man's outpainting will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)}.")

        state.job_count = batch_count

        for i in range(batch_count):
            p.init_images = [work[i]]
            p.image_mask = work_mask[i]
            p.latent_mask = work_latent_mask[i]

            state.job = f"Batch {i + 1} out of {batch_count}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images


        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)

        if opts.samples_save:
            images.save_image(combined_image, p.outpath_samples, "", initial_seed, p.prompt, opts.grid_format, info=initial_info)

        processed = Processed(p, [combined_image], initial_seed, initial_info)

        return processed

