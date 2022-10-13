import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images

class Script(scripts.Script):
    def title(self):
        return "To Infinity and Beyond"

    def ui(self, is_img2img):
        with gr.Row():
            n = gr.Textbox(label="n")
        with gr.Row():
            seed_type = gr.Radio(label='Seed Type', choices=["RandomSeed","RandomVariationSeed","RandomAllSeed"], value="RandomSeed", type="value", interactive=True)
        with gr.Row():
            instructions = gr.Textbox(label="For RandomVariationSeed and RandomAllSeed please enable the extras next to seed.", interactive=False)
        return [n, seed_type]

    def run(self, p, n, seed_type):
        if seed_type == "RandomVariationSeed":
            fixed_seed = p.seed
        for x in range(int(n)):
            if seed_type == "RandomVariationSeed":
                p.subseed = -1
            elif seed_type == "RandomAllSeed":
                p.seed = -1
                p.subseed = -1
            else:
                p.seed = -1
            
            proc = process_images(p)
            image = proc.images
        return Processed(p, image, p.seed, proc.info)