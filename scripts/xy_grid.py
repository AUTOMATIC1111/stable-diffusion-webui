from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
from io import StringIO
from PIL import Image
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images, paths, sd_samplers, processing, sd_models, sd_vae
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import glob
import os
import re

from modules.ui_components import ToolButton

fill_values_symbol = "\U0001f4d2"  # 📒


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []

    # Initally grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_name = sampler_name


def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise RuntimeError(f"Unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    modules.sd_models.reload_model_weights(shared.sd_model, info)


def confirm_checkpoints(p, xs):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x


def apply_upscale_latent_space(p, x, xs):
    if x.lower().strip() != '0':
        opts.data["use_scale_latent_for_hires_fix"] = True
    else:
        opts.data["use_scale_latent_for_hires_fix"] = False


def find_vae(name: str):
    if name.lower() in ['auto', 'automatic']:
        return modules.sd_vae.unspecified
    if name.lower() == 'none':
        return None
    else:
        choices = [x for x in sorted(modules.sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            print(f"No VAE found for {name}; using automatic")
            return modules.sd_vae.unspecified
        else:
            return modules.sd_vae.vae_dict[choices[0]]


def apply_vae(p, x, xs):
    modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))


def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles = x.split(',')


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


class AxisOption:
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True

class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    AxisOption("Prompt order", str_permutations, apply_order, format_value=format_value_join_list),
    AxisOptionTxt2Img("Sampler", str, apply_sampler, format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOptionImg2Img("Sampler", str, apply_sampler, format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img]),
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value=format_value, confirm=confirm_checkpoints, cost=1.0, choices=lambda: list(sd_models.checkpoints_list)),
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    AxisOption("Eta", float, apply_field("eta")),
    AxisOption("Clip skip", int, apply_clip_skip),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOptionImg2Img("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: list(sd_vae.vae_dict)),
    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
]


def draw_xy_grid(p, xs, ys, x_labels, y_labels, cell, draw_legend, include_lone_images, swap_axes_processing_order):
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]

    # Temporary list of all the images that are generated to be populated into the grid.
    # Will be filled with empty images for any individual step that fails to process properly
    image_cache = [None] * (len(xs) * len(ys))

    processed_result = None
    cell_mode = "P"
    cell_size = (1, 1)

    state.job_count = len(xs) * len(ys) * p.n_iter

    def process_cell(x, y, ix, iy):
        nonlocal image_cache, processed_result, cell_mode, cell_size

        state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

        processed: Processed = cell(x, y)

        try:
            # this dereference will throw an exception if the image was not processed
            # (this happens in cases such as if the user stops the process from the UI)
            processed_image = processed.images[0]

            if processed_result is None:
                # Use our first valid processed result as a template container to hold our full results
                processed_result = copy(processed)
                cell_mode = processed_image.mode
                cell_size = processed_image.size
                processed_result.images = [Image.new(cell_mode, cell_size)]

            image_cache[ix + iy * len(xs)] = processed_image
            if include_lone_images:
                processed_result.images.append(processed_image)
                processed_result.all_prompts.append(processed.prompt)
                processed_result.all_seeds.append(processed.seed)
                processed_result.infotexts.append(processed.infotexts[0])
        except:
            image_cache[ix + iy * len(xs)] = Image.new(cell_mode, cell_size)

    if swap_axes_processing_order:
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                process_cell(x, y, ix, iy)
    else:
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                process_cell(x, y, ix, iy)

    if not processed_result:
        print("Unexpected error: draw_xy_grid failed to return even a single processed image")
        return Processed(p, [])

    grid = images.image_grid(image_cache, rows=len(ys))
    if draw_legend:
        grid = images.draw_grid_annotations(grid, cell_size[0], cell_size[1], hor_texts, ver_texts)

    processed_result.images[0] = grid

    return processed_result


class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.vae = opts.sd_vae
  
    def __exit__(self, exc_type, exc_value, tb):
        opts.data["sd_vae"] = self.vae
        modules.sd_models.reload_model_weights()
        modules.sd_vae.reload_vae_weights()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")


class Script(scripts.Script):
    def title(self):
        return "X/Y plot"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]

        with gr.Row():
            with gr.Column(scale=19):
                with gr.Row():
                    x_type = gr.Dropdown(label="X type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[1].label, type="index", elem_id=self.elem_id("x_type"))
                    x_values = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"))
                    fill_x_button = ToolButton(value=fill_values_symbol, elem_id="xy_grid_fill_x_tool_button", visible=False)

                with gr.Row():
                    y_type = gr.Dropdown(label="Y type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                    y_values = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"))
                    fill_y_button = ToolButton(value=fill_values_symbol, elem_id="xy_grid_fill_y_tool_button", visible=False)

        with gr.Row(variant="compact", elem_id="axis_options"):
            draw_legend = gr.Checkbox(label='Draw legend', value=True, elem_id=self.elem_id("draw_legend"))
            include_lone_images = gr.Checkbox(label='Include Separate Images', value=False, elem_id=self.elem_id("include_lone_images"))
            no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))
            swap_axes_button = gr.Button(value="Swap axes", elem_id="xy_grid_swap_axes_button")

        def swap_axes(x_type, x_values, y_type, y_values):
            return self.current_axis_options[y_type].label, y_values, self.current_axis_options[x_type].label, x_values

        swap_args = [x_type, x_values, y_type, y_values]
        swap_axes_button.click(swap_axes, inputs=swap_args, outputs=swap_args)

        def fill(x_type):
            axis = self.current_axis_options[x_type]
            return ", ".join(axis.choices()) if axis.choices else gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type], outputs=[x_values])
        fill_y_button.click(fn=fill, inputs=[y_type], outputs=[y_values])

        def select_axis(x_type):
            return gr.Button.update(visible=self.current_axis_options[x_type].choices is not None)

        x_type.change(fn=select_axis, inputs=[x_type], outputs=[fill_x_button])
        y_type.change(fn=select_axis, inputs=[y_type], outputs=[fill_y_button])

        return [x_type, x_values, y_type, y_values, draw_legend, include_lone_images, no_fixed_seeds]

    def run(self, p, x_type, x_values, y_type, y_values, draw_legend, include_lone_images, no_fixed_seeds):
        if not no_fixed_seeds:
            modules.processing.fix_seed(p)

        if not opts.return_grid:
            p.batch_size = 1

        def process_axis(opt, vals):
            if opt.label == 'Nothing':
                return [0]

            valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals)))]

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2))+1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end   = int(mc.group(2))
                        num   = int(mc.group(3)) if mc.group(3) is not None else 1
                        
                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []

                for val in valslist:
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end   = float(mc.group(2))
                        num   = int(mc.group(3)) if mc.group(3) is not None else 1
                        
                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))

            valslist = [opt.type(x) for x in valslist]

            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)

            return valslist

        x_opt = self.current_axis_options[x_type]
        xs = process_axis(x_opt, x_values)

        y_opt = self.current_axis_options[y_type]
        ys = process_axis(y_opt, y_values)

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['Seed', 'Var. seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)

        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs)
        else:
            total_steps = p.steps * len(xs) * len(ys)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            if x_opt.label == "Hires steps":
                total_steps += sum(xs) * len(ys)
            elif y_opt.label == "Hires steps":
                total_steps += sum(ys) * len(xs)
            elif p.hr_second_pass_steps:
                total_steps += p.hr_second_pass_steps * len(xs) * len(ys)
            else:
                total_steps *= 2

        total_steps *= p.n_iter

        image_cell_count = p.n_iter * p.batch_size
        cell_console_text = f"; {image_cell_count} images per cell" if image_cell_count > 1 else ""
        print(f"X/Y plot will create {len(xs) * len(ys) * image_cell_count} images on a {len(xs)}x{len(ys)} grid{cell_console_text}. (Total steps to process: {total_steps})")
        shared.total_tqdm.updateTotal(total_steps)

        grid_infotext = [None]

        # If one of the axes is very slow to change between (like SD model
        # checkpoint), then make sure it is in the outer iteration of the nested
        # `for` loop.
        swap_axes_processing_order = x_opt.cost > y_opt.cost

        def cell(x, y):
            if shared.state.interrupted:
                return Processed(p, [], p.seed, "")

            pc = copy(p)
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)

            res = process_images(pc)

            if grid_infotext[0] is None:
                pc.extra_generation_params = copy(pc.extra_generation_params)

                if x_opt.label != 'Nothing':
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])

                if y_opt.label != 'Nothing':
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])

                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)

            return res

        with SharedSettingsStackHelper():
            processed = draw_xy_grid(
                p,
                xs=xs,
                ys=ys,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images,
                swap_axes_processing_order=swap_axes_processing_order
            )

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "xy_grid", info=grid_infotext[0], extension=opts.grid_format, prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        return processed
