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


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_pocessed = None

    state.job_count = len(xs) * len(ys)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed = cell(x, y)
            if first_pocessed is None:
                first_pocessed = processed

            res.append(processed.images[0])

    grid = images.image_grid(res, rows=len(ys))
    grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    first_pocessed.images = [grid]

    return first_pocessed


class Script(scripts.Script):
    def title(self):
        return "Advanced prompt matrix"

    def ui(self, is_img2img):
            return None

    def run(self, p):
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
