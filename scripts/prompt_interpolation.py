import modules.scripts as scripts
import gradio as gr
import random
import os
from PIL import Image

from modules import images
from modules.processing import process_images
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


def process(p, prompt1, prompt2, n_images):
    first_processed = None
    processed_images = []

    for i in range(p.batch_size * p.n_iter):
        processed_images.append([])
    
    state.job_count = n_images * p.n_iter

    for i in range(n_images):
        state.job = f"interpolation: {i + 1} out of {n_images}"
        interpolation = 0.5 if n_images == 1 else i / (n_images - 1)
        p.prompt = f"{prompt1} :{1 - interpolation} AND {prompt2} :{interpolation}"
        processed = process_images(p)
        
        if first_processed is None:
            first_processed = processed

        for i, img in enumerate(processed.images):
            processed_images[i].append(img)

    return first_processed, processed_images


class Script(scripts.Script):

    def title(self):
        return "Prompts interpolation"


    def show(self, is_img2img):
        return True


    def ui(self, is_img2img):
        prompt2 = gr.TextArea(label="Interpolation prompt")
        n_images = gr.Slider(minimum=1, maximum=128, step=1, value=1, label="Number of images")
        make_a_gif = gr.Checkbox(label="Make a gif", value=True)
        duration = gr.Slider(minimum=1, maximum=1000, step=1, value=100, label="Duration of images (ms)", visible=True)
        make_a_gif.change(fn=lambda x: gr.update(visible=x), inputs=[make_a_gif], outputs=[duration])
        return [prompt2, n_images, make_a_gif, duration]


    def run(self, p, prompt2, n_images, make_a_gif, duration):
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))
        
        p.do_not_save_grid = True
        prompt1 = p.prompt
        
        processed, processed_images = process(p, prompt1, prompt2, n_images)

        p.prompt_for_display = processed.prompt = f"{prompt1} AND {prompt2}"
        processed_images_flattened = []
        
        for row in processed_images:
            processed_images_flattened += row
        
        if len(processed_images_flattened) == 1:
            processed.images = processed_images_flattened
        else:
            processed.images = [images.image_grid(processed_images_flattened, rows=p.batch_size * p.n_iter)] \
                + processed_images_flattened
        
        if make_a_gif or opts.grid_save:
            (fullfn, _) = images.save_image(processed.images[0], p.outpath_grids, "grid",
                prompt=p.prompt_for_display, seed=processed.seed, grid=True, p=p)
        
        if make_a_gif:
            for i, row in enumerate(processed_images):
                fullfn = fullfn[:fullfn.rfind(".")] + "_" + str(i) + ".gif"
                # since there is no option for saving gif images in images.save_image(), I had to
                # do it from scratch, maybe it can be improved in the future
                processed_images[i][0].save(fullfn, save_all=True,
                    append_images=processed_images[i][1:], optimize=False, duration=duration, loop=0)
        
        return processed