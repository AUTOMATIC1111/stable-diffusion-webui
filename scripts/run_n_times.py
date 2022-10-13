import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
import glob
import random

def replace_wildcard(chunk):
    if " " not in chunk:
        file_dir = os.path.dirname(os.path.realpath("__file__"))
        replacement_file = os.path.join(file_dir, f"cfg/promptgen/{chunk}.csv")
        if os.path.exists(replacement_file):
            with open(replacement_file, "r", encoding="utf8", newline='') as f:
                lines = f.readlines()
                stripped = []
                for line in lines:
                    stripped.append(line.strip())                        
                stripped.remove('name,blank,blank2')
                #print(stripped)
                return random.choice(stripped).replace(",,","")
    return chunk


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
            
            p.prompt = "".join(replace_wildcard(chunk) for chunk in p.prompt.split("__"))
            
            proc = process_images(p)
            image = proc.images
        return Processed(p, image, p.seed, proc.info)