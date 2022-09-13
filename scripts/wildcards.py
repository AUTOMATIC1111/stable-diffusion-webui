import math
import os
import sys
import traceback
import random

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Wildcards"

    def ui(self, is_img2img):
        same_seed = gr.Checkbox(label='Use same seed for each image', value=False)

        return [same_seed]

    def run(self, p, same_seed):
        def replace_wildcard(chunk):
            if " " not in chunk:
                file_dir = os.path.dirname(os.path.realpath("__file__"))
                replacement_file = os.path.join(file_dir, f"scripts/wildcards/{chunk}.txt")
                if os.path.exists(replacement_file):
                    with open(replacement_file, encoding="utf8") as f:
                        return random.choice(f.read().splitlines())
            return chunk
        
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        all_prompts = ["".join(replace_wildcard(chunk) for chunk in original_prompt.split("__")) for _ in range(p.batch_size * p.n_iter)]

        print(f"Will process {p.batch_size * p.n_iter} images in {p.n_iter} batches.")

        p.do_not_save_grid = True

        state.job_count = p.n_iter
        p.n_iter = 1

        images = []
        for batch_no in range(state.job_count):
            state.job = f"{batch_no+1} out of {state.job_count}"
            p.prompt = all_prompts[batch_no*p.batch_size:(batch_no+1)*p.batch_size]
            proc = process_images(p)
            images += proc.images
            if not same_seed:
                p.seed += 1

        return Processed(p, images, p.seed, "")