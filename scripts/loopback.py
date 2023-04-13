import math

import gradio as gr
import modules.scripts as scripts
from modules import deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state


class Script(scripts.Script):
    def title(self):
        return "Loopback"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):        
        loops = gr.Slider(minimum=1, maximum=32, step=1, label='Loops', value=4, elem_id=self.elem_id("loops"))
        final_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='Final denoising strength', value=0.5, elem_id=self.elem_id("final_denoising_strength"))
        denoising_curve = gr.Dropdown(label="Denoising strength curve", choices=["Aggressive", "Linear", "Lazy"], value="Linear")
        append_interrogation = gr.Dropdown(label="Append interrogated prompt at each iteration", choices=["None", "CLIP", "DeepBooru"], value="None")

        return [loops, final_denoising_strength, denoising_curve, append_interrogation]

    def run(self, p, loops, final_denoising_strength, denoising_curve, append_interrogation):
        processing.fix_seed(p)
        batch_count = p.n_iter
        p.extra_generation_params = {
            "Final denoising strength": final_denoising_strength,
            "Denoising curve": denoising_curve
        }

        p.batch_size = 1
        p.n_iter = 1

        info = None
        initial_seed = None
        initial_info = None
        initial_denoising_strength = p.denoising_strength

        grids = []
        all_images = []
        original_init_image = p.init_images
        original_prompt = p.prompt
        original_inpainting_fill = p.inpainting_fill
        state.job_count = loops * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        def calculate_denoising_strength(loop):
            strength = initial_denoising_strength

            if loops == 1:
                return strength

            progress = loop / (loops - 1)
            if denoising_curve == "Aggressive":
                strength = math.sin((progress) * math.pi * 0.5)
            elif denoising_curve == "Lazy":
                strength = 1 - math.cos((progress) * math.pi * 0.5)
            else:
                strength = progress

            change = (final_denoising_strength - initial_denoising_strength) * strength
            return initial_denoising_strength + change

        history = []

        for n in range(batch_count):
            # Reset to original init image at the start of each batch
            p.init_images = original_init_image

            # Reset to original denoising strength
            p.denoising_strength = initial_denoising_strength

            last_image = None

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True

                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections

                if append_interrogation != "None":
                    p.prompt = original_prompt + ", " if original_prompt != "" else ""
                    if append_interrogation == "CLIP":
                        p.prompt += shared.interrogator.interrogate(p.init_images[0])
                    elif append_interrogation == "DeepBooru":
                        p.prompt += deepbooru.model.tag(p.init_images[0])

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

                processed = processing.process_images(p)

                # Generation cancelled.
                if state.interrupted:
                    break

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                p.seed = processed.seed + 1
                p.denoising_strength = calculate_denoising_strength(i + 1)
                
                if state.skipped:
                    break

                last_image = processed.images[0]
                p.init_images = [last_image]
                p.inpainting_fill = 1 # Set "masked content" to "original" for next loop.

                if batch_count == 1:
                    history.append(last_image)
                    all_images.append(last_image)

            if batch_count > 1 and not state.skipped and not state.interrupted:
                history.append(last_image)
                all_images.append(last_image)

            p.inpainting_fill = original_inpainting_fill
                
            if state.interrupted:
                    break

        if len(history) > 1:
            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

            if opts.return_grid:
                grids.append(grid)
                
        all_images = grids + all_images

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
