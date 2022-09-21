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
import modules.shared as shared
import glob

class Script(scripts.Script):
    def title(self):
        return "Wildcards CSV"

    def ui(self, is_img2img):
        with gr.Row():
            txt_list = str(glob.glob(r'cfg/promptgen\*.csv')).replace(".csv","").replace("cfg/promptgen\\\\","")
            dummy = gr.Textbox(label='Wildcard List', value=f'{txt_list}', interactive=False, lines=3)
        with gr.Row():
            seed_type = gr.Radio(label='Seed Type', choices=["SameSeed","IncSeed","RandomSeed"], value="IncSeed", type="value", interactive=True)
        return [seed_type, dummy]

    def run(self, p, seed_type, dummy):
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
                        print(stripped)
                        return random.choice(stripped).replace(",,","")
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
            if seed_type == "IncSeed":
                p.seed += 1
            if seed_type == "RandomSeed":
                p.seed = int(random.randrange(4294967294))

        return Processed(p, images, p.seed, "")
