from collections import namedtuple
from copy import copy
import random

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    p.prompt = p.prompt.replace(xs[0], x)


samplers_dict = {}
for i, sampler in enumerate(modules.sd_samplers.samplers):
    samplers_dict[sampler.name.lower()] = i
    for alias in sampler.aliases:
        samplers_dict[alias.lower()] = i


def apply_sampler(p, x, xs):
    sampler_index = samplers_dict.get(x.lower(), None)
    print(x, sampler_index)
    if sampler_index is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_index = sampler_index


def format_value_add_label(p, opt, x):
    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    return x


AxisOption = namedtuple("AxisOption", ["label", "type", "apply", "format_value"])
AxisOptionImg2Img = namedtuple("AxisOptionImg2Img", ["label", "type", "apply", "format_value"])


axis_options = [
    AxisOption("Seed", int, apply_field("seed"), format_value_add_label),
    AxisOption("Steps", int, apply_field("steps"), format_value_add_label),
    AxisOption("CFG Scale", float, apply_field("cfg_scale"), format_value_add_label),
    AxisOption("Prompt S/R", str, apply_prompt, format_value),
    AxisOption("Sampler", str, apply_prompt, format_value),
    AxisOptionImg2Img("Denoising", float, apply_field("denoising_strength"), format_value_add_label) #  as it is now all AxisOptionImg2Img items must go after AxisOption ones
]


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_pocessed = None

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
        return "X/Y plot"

    def ui(self, is_img2img):
        current_axis_options = [x for x in axis_options if type(x) == AxisOption or type(x) == AxisOptionImg2Img and is_img2img]

        with gr.Row():
            x_type = gr.Dropdown(label="X type", choices=[x.label for x in current_axis_options], value=current_axis_options[0].label, visible=False, type="index", elem_id="x_type")
            x_values = gr.Textbox(label="X values", visible=False, lines=1)

        with gr.Row():
            y_type = gr.Dropdown(label="Y type", choices=[x.label for x in current_axis_options], value=current_axis_options[1].label, visible=False, type="index", elem_id="y_type")
            y_values = gr.Textbox(label="Y values", visible=False, lines=1)

        return [x_type, x_values, y_type, y_values]

    def run(self, p, x_type, x_values, y_type, y_values):
        p.seed = int(random.randrange(4294967294) if p.seed == -1 else p.seed)

        def process_axis(opt, vals):
            valslist = [x.strip() for x in vals.split(",")]

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    if "-" in val:
                        s = val.split("-")
                        start = int(s[0])
                        end = int(s[1])+1
                        step = 1 if len(s) < 3 else int(s[2])
                        valslist_ext += list(range(start, end, step))
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

        images.save_image(processed.images[0], p.outpath_grids, "xy_grid", prompt=p.prompt, seed=processed.seed)

        return processed
