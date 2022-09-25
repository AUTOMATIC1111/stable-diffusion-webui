import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state

from braceexpand import braceexpand


class Script(scripts.Script):
    def title(self):
        return "Prompts with brace expand"

    def ui(self, is_img2img):
        return []

    def run(self, p):
        lines = list(braceexpand(p.prompt))

        img_count = len(lines) * p.n_iter
        batch_count = math.ceil(img_count / p.batch_size)
        loop_count = math.ceil(batch_count / p.n_iter)
        print(f"Will process {img_count} images in {batch_count} batches.")

        p.do_not_save_grid = True

        state.job_count = batch_count

        images = []
        for loop_no in range(loop_count):
            state.job = f"{loop_no + 1} out of {loop_count}"
            p.prompt = lines[loop_no*p.batch_size:(loop_no+1)*p.batch_size] * p.n_iter
            proc = process_images(p)
            images += proc.images

        return Processed(p, images, p.seed, "")
