from collections import namedtuple
from copy import copy
import random

import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers
import re


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


samplers_dict = {}
for i, sampler in enumerate(modules.sd_samplers.samplers):
    samplers_dict[sampler.name.lower()] = i
    for alias in sampler.aliases:
        samplers_dict[alias.lower()] = i


def apply_sampler(p, x, xs):
    sampler_index = samplers_dict.get(x.lower(), None)
    if sampler_index is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_index = sampler_index


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return x

def do_nothing(p, x, xs):
    pass

def format_nothing(p, opt, x):
    return ""


AxisOption = namedtuple("AxisOption", ["label", "type", "apply", "format_value"])
AxisOptionImg2Img = namedtuple("AxisOptionImg2Img", ["label", "type", "apply", "format_value"])


axis_options = [
    AxisOption("Nothing", str, do_nothing, format_nothing),
    AxisOption("Seed", int, apply_field("seed"), format_value_add_label),
    AxisOption("Var. seed", int, apply_field("subseed"), format_value_add_label),
    AxisOption("Var. strength", float, apply_field("subseed_strength"), format_value_add_label),
    AxisOption("Steps", int, apply_field("steps"), format_value_add_label),
    AxisOption("CFG Scale", float, apply_field("cfg_scale"), format_value_add_label),
    AxisOption("Prompt S/R", str, apply_prompt, format_value),
    AxisOption("Sampler", str, apply_sampler, format_value),
    AxisOptionImg2Img("Denoising", float, apply_field("denoising_strength"), format_value_add_label), #  as it is now all AxisOptionImg2Img items must go after AxisOption ones
]


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


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

class Script(scripts.Script):
    def title(self):
        return "X/Y plot"

    def ui(self, is_img2img):
        current_axis_options = [x for x in axis_options if type(x) == AxisOption or type(x) == AxisOptionImg2Img and is_img2img]

        with gr.Row():
            x_type = gr.Dropdown(label="X type", choices=[x.label for x in current_axis_options], value=current_axis_options[1].label, visible=False, type="index", elem_id="x_type")
            x_values = gr.Textbox(label="X values", visible=False, lines=1)

        with gr.Row():
            y_type = gr.Dropdown(label="Y type", choices=[x.label for x in current_axis_options], value=current_axis_options[4].label, visible=False, type="index", elem_id="y_type")
            y_values = gr.Textbox(label="Y values", visible=False, lines=1)

        return [x_type, x_values, y_type, y_values]

    def run(self, p, x_type, x_values, y_type, y_values):
        modules.processing.fix_seed(p)
        p.batch_size = 1
        p.batch_count = 1

        def process_axis(opt, vals):
            valslist = [x.strip() for x in vals.split(",")]

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    m = re_range.fullmatch(val)
                    if m is not None:

                        start = int(m.group(1))
                        end = int(m.group(2))+1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []

                for val in valslist:
                    m = re_range_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += np.arange(start, end + step, step).tolist()
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext

            valslist = [opt.type(x) for x in valslist]

            return valslist

        x_opt = axis_options[x_type]
        xs = process_axis(x_opt, x_values)

        y_opt = axis_options[y_type]
        ys = process_axis(y_opt, y_values)

        def cell(x, y):
            pc = copy(p)
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)

            return process_images(pc)

        processed = draw_xy_grid(
            xs=xs,
            ys=ys,
            x_label=lambda x: x_opt.format_value(p, x_opt, x),
            y_label=lambda y: y_opt.format_value(p, y_opt, y),
            cell=cell
        )

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "xy_grid", prompt=p.prompt, seed=processed.seed, p=p)

        return processed
