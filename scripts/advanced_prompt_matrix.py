import math
from collections import namedtuple
from copy import copy
import random
import re

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers

class Script(scripts.Script):
    def title(self):
        return "Advanced prompt matrix"

    def ui(self, is_img2img):
        dummy = gr.Checkbox(label="Usage: a <corgi|cat> wearing <goggles|a hat>")
        return [dummy]


    def run(self, p, dummy):
        modules.processing.fix_seed(p)

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        matrix_count = 0
        prompt_matrix_parts = []
        for data in re.finditer(r'(<([^>]+)>)', original_prompt):
            if data:
                matrix_count += 1
                span = data.span(1)
                items = data.group(2).split("|")
                prompt_matrix_parts.extend(items)

        all_prompts = [original_prompt]
        while True:
            found_matrix = False
            for this_prompt in all_prompts:
                for data in re.finditer(r'(<([^>]+)>)', this_prompt):
                    if data:
                        found_matrix = True
                        # Remove last prompt as it has a found_matrix
                        all_prompts.remove(this_prompt)
                        span = data.span(1)
                        items = data.group(2).split("|")
                        for item in items:
                            new_prompt = this_prompt[:span[0]] + item.strip() + this_prompt[span[1]:]
                            all_prompts.append(new_prompt.strip())
                    break
                if found_matrix:
                    break
            if not found_matrix:
                break

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

        p.prompt = all_prompts
        p.seed = [p.seed for _ in all_prompts]
        p.prompt_for_display = original_prompt
        processed = process_images(p)

        return processed
