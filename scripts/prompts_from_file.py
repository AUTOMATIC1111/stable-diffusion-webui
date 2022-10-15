import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):
    def title(self):
        return "Prompts from file or textbox"

    def ui(self, is_img2img):
        # This checkbox would look nicer as two tabs, but there are two problems:
        # 1) There is a bug in Gradio 3.3 that prevents visibility from working on Tabs
        # 2) Even with Gradio 3.3.1, returning a control (like Tabs) that can't be used as input
        #    causes a AttributeError: 'Tabs' object has no attribute 'preprocess' assert,
        #    due to the way Script assumes all controls returned can be used as inputs.
        # Therefore, there's no good way to use grouping components right now,
        # so we will use a checkbox! :)
        checkbox_txt = gr.Checkbox(label="Show Textbox", value=False)
        file = gr.File(label="File with inputs", type='bytes')
        prompt_txt = gr.TextArea(label="Prompts")
        checkbox_txt.change(fn=lambda x: [gr.File.update(visible = not x), gr.TextArea.update(visible = x)], inputs=[checkbox_txt], outputs=[file, prompt_txt])
        return [checkbox_txt, file, prompt_txt]

    def process_string_tag(self, tag):
        return tag[1:-2]

    def process_int_tag(self, tag):
        return int(tag)

    def process_float_tag(self, tag):
        return float(tag)

    def process_boolean_tag(self, tag):
        return True if (tag == "true") else False

    prompt_tags = {
        "sd_model": None,
        "outpath_samples": process_string_tag,
        "outpath_grids": process_string_tag,
        "prompt_for_display": process_string_tag,
        "prompt": process_string_tag,
        "negative_prompt": process_string_tag,
        "styles": process_string_tag,
        "seed": process_int_tag,
        "subseed_strength": process_float_tag,
        "subseed": process_int_tag,
        "seed_resize_from_h": process_int_tag,
        "seed_resize_from_w": process_int_tag,
        "sampler_index": process_int_tag,
        "batch_size": process_int_tag,
        "n_iter": process_int_tag,
        "steps": process_int_tag,
        "cfg_scale": process_float_tag,
        "width": process_int_tag,
        "height": process_int_tag,
        "restore_faces": process_boolean_tag,
        "tiling": process_boolean_tag,
        "do_not_save_samples": process_boolean_tag,
        "do_not_save_grid": process_boolean_tag
    }

    def on_show(self, checkbox_txt, file, prompt_txt):
        return [ gr.Checkbox.update(visible = True), gr.File.update(visible = not checkbox_txt), gr.TextArea.update(visible = checkbox_txt) ]

    def run(self, p, checkbox_txt, data: bytes, prompt_txt: str):
        if (checkbox_txt):
            lines = [x.strip() for x in prompt_txt.splitlines()]
        else:
            lines = [x.strip() for x in data.decode('utf8', errors='ignore').split("\n")]
        lines = [x for x in lines if len(x) > 0]

        img_count = len(lines) * p.n_iter
        batch_count = math.ceil(img_count / p.batch_size)
        loop_count = math.ceil(batch_count / p.n_iter)
        # These numbers no longer accurately reflect the total images and number of batches
        print(f"Will process {img_count} images in {batch_count} batches.")

        p.do_not_save_grid = True

        state.job_count = batch_count

        images = []
        for loop_no in range(loop_count):
            state.job = f"{loop_no + 1} out of {loop_count}"
            # The following line may need revising to remove batch_size references
            current_line = lines[loop_no*p.batch_size:(loop_no+1)*p.batch_size] * p.n_iter

            # If the current line has no tags, parse the whole line as a prompt, else parse each tag
            if(current_line[0][:2] != "--"):
                p.prompt = current_line
            else:
                tokenized_line = current_line[0].split("--")

                for tag in tokenized_line:
                    tag_split = tag.split(" ", 1)
                    if(tag_split[0] != ''):
                        value_func = self.prompt_tags.get(tag_split[0], None)
                        if(value_func != None):
                            value = value_func(self, tag_split[1])
                            setattr(p, tag_split[0], value)
                        else:
                            print(f"Unknown option \"{tag_split}\"")

            proc = process_images(p)
            images += proc.images

        return Processed(p, images, p.seed, "")
