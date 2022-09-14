import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Prompts from file"

    def ui(self, is_img2img):
        file = gr.File(label="File with inputs", type='bytes')

        return [file]

    def run(self, p, data: bytes):
        lines = [x.strip() for x in data.decode('utf8', errors='ignore').split("\n")]
        lines = [x for x in lines if len(x) > 0]

        batch_count = math.ceil(len(lines) / p.batch_size)
        print(f"Will process {len(lines) * p.n_iter} images in {batch_count * p.n_iter} batches.")

        p.do_not_save_grid = True

        state.job_count = batch_count

        images = []
        for batch_no in range(batch_count):
            state.job = f"{batch_no} out of {batch_count * p.n_iter}"
            p.prompt = lines[batch_no*p.batch_size:(batch_no+1)*p.batch_size] * p.n_iter
            proc = process_images(p)
            images += proc.images

        return Processed(p, images, p.seed, "")
