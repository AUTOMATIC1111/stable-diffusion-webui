import math
from collections import namedtuple
from copy import copy
import random

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
        return "Prompt matrix"

    def ui(self, is_img2img):
        put_at_start = gr.Checkbox(label='Put variable parts at start of prompt', value=False)

        return [put_at_start]

    def run(self, p, put_at_start):
        seed = int(random.randrange(4294967294) if p.seed == -1 else p.seed)

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        all_prompts = []
        prompt_matrix_parts = original_prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [text.strip().strip(',') for n, text in enumerate(prompt_matrix_parts[1:]) if combination_num & (1 << n)]

            if put_at_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append(", ".join(selected_prompts))

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

        p.prompt = all_prompts
        p.prompt_for_display = original_prompt
        p.seed = len(all_prompts) * [seed]
        processed = process_images(p)

        grid = images.image_grid(processed.images, p.batch_size, rows=1 << ((len(prompt_matrix_parts) - 1) // 2))
        grid = images.draw_prompt_matrix(grid, p.width, p.height, prompt_matrix_parts)
        processed.images.insert(0, grid)

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "prompt_matrix", prompt=original_prompt, seed=seed)

        return processed
