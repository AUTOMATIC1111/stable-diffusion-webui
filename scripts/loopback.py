import numpy as np
from tqdm import trange

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Loopback"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):        
        loops = gr.Slider(minimum=1, maximum=32, step=1, label='Loops', value=4, elem_id=self.elem_id("loops"))
        denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1, elem_id=self.elem_id("denoising_strength_change_factor"))

        return [loops, denoising_strength_change_factor]

    def run(self, p, loops, denoising_strength_change_factor):
        processing.fix_seed(p)
        batch_count = p.n_iter
        p.extra_generation_params = {
            "Denoising strength change factor": denoising_strength_change_factor,
        }

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        grids = []
        all_images = []
        original_init_image = p.init_images
        state.job_count = loops * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        for n in range(batch_count):
            history = []

            # Reset to original init image at the start of each batch
            p.init_images = original_init_image

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True

                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

                processed = processing.process_images(p)

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                init_img = processed.images[0]

                p.init_images = [init_img]
                p.seed = processed.seed + 1
                p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
                history.append(processed.images[0])

            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

            grids.append(grid)
            all_images += history

        if opts.return_grid:
            all_images = grids + all_images

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
