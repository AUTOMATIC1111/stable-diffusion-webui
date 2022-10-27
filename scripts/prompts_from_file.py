import copy
import math
import os
import sys
import traceback
import shlex

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
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


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        tag = arg[2:]

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        val = args[pos+1]

        res[tag] = func(val)

        pos += 2

    return res


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

    def on_show(self, checkbox_txt, file, prompt_txt):
        return [ gr.Checkbox.update(visible = True), gr.File.update(visible = not checkbox_txt), gr.TextArea.update(visible = checkbox_txt) ]

    def run(self, p, checkbox_txt, data: bytes, prompt_txt: str):
        if checkbox_txt:
            lines = [x.strip() for x in prompt_txt.splitlines()]
        else:
            lines = [x.strip() for x in data.decode('utf8', errors='ignore').split("\n")]
        lines = [x for x in lines if len(x) > 0]

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    print(f"Error parsing line [line] as commandline:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            n_iter = args.get("n_iter", 1)
            if n_iter != 1:
                job_count += n_iter
            else:
                job_count += 1

            jobs.append(args)

        print(f"Will process {len(lines)} lines in {job_count} jobs.")
        state.job_count = job_count

        images = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images

        return Processed(p, images, p.seed, "")
