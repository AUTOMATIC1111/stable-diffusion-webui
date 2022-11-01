import math
import os
import sys
import traceback
import random
import re

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state

from copy import copy
from tqdm import tqdm
from aitextgen import aitextgen

class Script(scripts.Script):
    def title(self):
        return "Prompts from MagicPrompt"

    def ui(self, is_img2img):
        prompt_length = gr.Slider(label='Max prompt length', value=100, minimum=1, maximum=150, step=1)
        temperature_value = gr.Slider(label='Temperature', value=0.7, minimum=0.1, maximum=3.0, step=0.05)
        return [prompt_length, temperature_value]

    def run(self, p, prompt_length, temperature_value):
        processing.fix_seed(p)
        batch_count = p.n_iter

        gpt2model = os.path.join('repositories', 'MagicPrompt-Stable-Diffusion')
        ai        = aitextgen(model_folder=gpt2model, tokenizer_file=os.path.join(gpt2model, 'tokenizer.json'))

        # Batch all of these first, to retain the same seed as the main image processor
        print(f"\nGenerating prompts...")
        all_prompts = ai.generate(n=int(batch_count * 1.2), prompt=p.prompt, max_length=prompt_length, temperature=float(temperature_value), seed=p.seed, return_as_list=True)
        all_prompts = [self.clean_up_ai_prompt(mp) for mp in all_prompts]

        # ai.generate may give us less than what we want, so we just ask for 1.2x more and drop the rest
        all_prompts = all_prompts[0:batch_count]

        all_images = []
        all_infos  = []
        state.job_count = batch_count

        for n in range(batch_count):
            if shared.state.interrupted:
                break

            shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            pc = copy(p)
            pc.n_iter = 1
            pc.do_not_save_grid = True

            pc.prompt = all_prompts[n]
            print(f"\n\nGenerated prompt: {pc.prompt}")

            processed = processing.process_images(pc)

            all_images.extend(processed.images)
            all_infos.append(processed.info)

        rows = None if p.batch_size == 1 else batch_count
        grid = images.image_grid(all_images, rows=rows)
        if opts.grid_save:
            images.save_image(grid, p.outpath_grids, "grid", p.seed, all_prompts[0], opts.grid_format, info="\n\n".join(all_infos), short_filename=not opts.grid_extended_filename, grid=True, p=p)

        if opts.return_grid:
            all_images.insert(0, grid)

        processed = Processed(p, all_images, p.seed, all_infos[0], p.subseed, all_prompts, infotexts=all_infos)

        return processed

    def clean_up_ai_prompt(self, prompt):
        prompt = prompt.translate( str.maketrans('{}', '()') ).strip()

        prompt = re.sub(r'^\W+|\W+$', '', prompt)   # useless non-word characters at the begin/end
        prompt = re.sub(r'\(\s+', '(', prompt)      # clean up whitespace in weighted parens
        prompt = re.sub(r'\s+\)', ')', prompt)
        prompt = re.sub(r'\b\s+\-\s+\b', '-', prompt)  # clean up whitespace in hyphens between words
        prompt = re.sub(r'\s*[,;|:\.]+\s*', ', ', prompt)  # other analogues to ', '
        prompt = re.sub(r'\s+_+\s+', ' ', prompt)   # useless underscores between phrases

        # Translate bangs into proper weight modifiers
        for match in re.findall(r'\b([\w\s\-]+)(\!+)', prompt):
            phrase     = match[0]
            full_match = match[0] + match[1]
            weight     = round( pow(1.1, len(match[1])), 2 )

            prompt = prompt.replace(full_match, f'({phrase}:{weight})')

        return prompt
