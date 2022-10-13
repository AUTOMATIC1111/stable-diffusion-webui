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
        return "To Infinity and Beyond (Wildcard Support)"

    def ui(self, is_img2img):
        with gr.Row():
            n = gr.Textbox(label="Images to generate")
        with gr.Row():
            seed_type = gr.Radio(label='Seed Type. (For RandomVariationSeed and RandomAllSeed please enable the extras next to seed.)', choices=["RandomSeed","RandomVariationSeed","RandomAllSeed"], value="RandomSeed", type="value", interactive=True)
        with gr.Row():
            wildcard_behaviour = gr.Radio(label='Wildcard behaviour. Randomise wildcards per image or batch.', choices=["Batch","Image"], value="Batch", type="value", interactive=True)
        with gr.Row():
            txt_list = str(glob.glob(r'cfg/promptgen\*.csv')).replace(".csv","").replace("cfg/promptgen\\\\","")
            dummy = gr.Textbox(label='Wildcard List', value=f'{txt_list}', interactive=False, lines=3)
        return [n, seed_type, wildcard_behaviour, dummy]

    def run(self, p, n, seed_type):
        original_prompt = p.prompt
        if wildcard_behaviour == "Batch":
            p.prompt = "".join(replace_wildcard(chunk) for chunk in original_prompt.split("__"))

        for x in range(int(n)):
            if seed_type == "RandomVariationSeed":
                p.subseed = -1
            elif seed_type == "RandomAllSeed":
                p.seed = -1
                p.subseed = -1
            else:
                p.seed = -1

            if wildcard_behaviour == "Image":
                p.prompt = "".join(replace_wildcard(chunk) for chunk in original_prompt.split("__"))

            proc = process_images(p)
            image = proc.images

        return Processed(p, image, p.seed, proc.info)