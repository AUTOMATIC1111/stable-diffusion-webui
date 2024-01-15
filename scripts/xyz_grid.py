# pylint: disable=unused-argument

import re
import csv
import random
from collections import namedtuple
from copy import copy
from itertools import permutations, chain
from io import StringIO
from PIL import Image
import numpy as np
import gradio as gr
from modules import shared, errors, scripts, images, sd_samplers, processing, sd_models, sd_vae
from modules.ui_components import ToolButton
import modules.ui_symbols as symbols


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)
    return fun


def apply_setting(field):
    def fun(p, x, xs):
        shared.opts.data[field] = x
    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        shared.log.warning(f"XYZ grid: prompt S/R did not find {xs[0]} in prompt or negative prompt.")
    else:
        p.prompt = p.prompt.replace(xs[0], x)
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []
    for token in x:
        token_order.append((p.prompt.find(token), token))
    token_order.sort(key=lambda t: t[0])
    prompt_parts = []
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        shared.log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.sampler_name = sampler_name

def apply_hr_sampler_name(p, x, xs):
    hr_sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if hr_sampler_name is None:
        shared.log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.hr_sampler_name = hr_sampler_name

def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            shared.log.warning(f"XYZ grid: unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    if x == shared.opts.sd_model_checkpoint:
        return
    info = sd_models.get_closet_checkpoint_match(x)
    if info is None:
        shared.log.warning(f"XYZ grid: apply checkpoint unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name


def apply_refiner(p, x, xs):
    if x == shared.opts.sd_model_refiner:
        return
    info = sd_models.get_closet_checkpoint_match(x)
    if info is None:
        shared.log.warning(f"XYZ grid: apply refiner unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_refiner, info)
        p.override_settings['sd_model_refiner'] = info.name


def apply_dict(p, x, xs):
    if x == shared.opts.sd_model_dict:
        return
    info_dict = sd_models.get_closet_checkpoint_match(x)
    info_ckpt = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
    if info_dict is None or info_ckpt is None:
        shared.log.warning(f"XYZ grid: apply dict unknown checkpoint: {x}")
    else:
        shared.opts.sd_model_dict = info_dict.name # this will trigger reload_model_weights via onchange handler
        p.override_settings['sd_model_checkpoint'] = info_ckpt.name
        p.override_settings['sd_model_dict'] = info_dict.name


def apply_clip_skip(p, x, xs):
    p.clip_skip = x
    shared.opts.data["clip_skip"] = x


def find_vae(name: str):
    if name.lower() in ['auto', 'automatic']:
        return sd_vae.unspecified
    if name.lower() == 'none':
        return None
    else:
        choices = [x for x in sorted(sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            shared.log.warning(f"No VAE found for {name}; using automatic")
            return sd_vae.unspecified
        else:
            return sd_vae.vae_dict[choices[0]]


def apply_vae(p, x, xs):
    sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))


def apply_styles(p: processing.StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(','))


def apply_upscaler(p: processing.StableDiffusionProcessingTxt2Img, opt, x):
    p.enable_hr = True
    p.hr_force = True
    p.denoising_strength = 0.0
    p.hr_upscaler = opt


def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    else:
        is_active = opt in ('true', 'yes', 'y', '1')
    p.restore_faces = is_active


def apply_override(field):
    def fun(p, x, xs):
        p.override_settings[field] = x
    return fun


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


def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()


class AxisOption:
    def __init__(self, label, tipe, apply, fmt=format_value_add_label, confirm=None, cost=0.0, choices=None):
        self.label = label
        self.type = tipe
        self.apply = apply
        self.format_value = fmt
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
    AxisOption("Nothing", str, do_nothing, fmt=format_nothing),
    AxisOption("Prompt S/R", str, apply_prompt, fmt=format_value),
    AxisOption("Model", str, apply_checkpoint, fmt=format_value, cost=1.0, choices=lambda: sorted(sd_models.checkpoints_list)),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['None'] + list(sd_vae.vae_dict)),
    AxisOption("Styles", str, apply_styles, choices=lambda: [s.name for s in shared.prompt_styles.styles.values()]),
    AxisOptionTxt2Img("Sampler", str, apply_sampler, fmt=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOptionImg2Img("Sampler", str, apply_sampler, fmt=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img]),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOption("Variation seed", int, apply_field("subseed")),
    AxisOption("Variation strength", float, apply_field("subseed_strength")),
    AxisOption("Clip skip", int, apply_clip_skip),
    AxisOption("Denoising strength", float, apply_field("denoising_strength")),
    AxisOption("Prompt order", str_permutations, apply_order, fmt=format_value_join_list),
    AxisOption("Model dictionary", str, apply_dict, fmt=format_value, cost=1.0, choices=lambda: ['None'] + list(sd_models.checkpoints_list)),
    AxisOptionImg2Img("Image mask weight", float, apply_field("inpainting_mask_weight")),
    AxisOption("[Postprocess] Upscaler", str, apply_upscaler, choices=lambda: [x.name for x in shared.sd_upscalers][1:]),
    AxisOption("[Postprocess] Face restore", str, apply_face_restore, fmt=format_value),
    AxisOption("[Sampler] Sigma min", float, apply_field("s_min")),
    AxisOption("[Sampler] Sigma max", float, apply_field("s_max")),
    AxisOption("[Sampler] Sigma tmin", float, apply_field("s_tmin")),
    AxisOption("[Sampler] Sigma tmax", float, apply_field("s_tmax")),
    AxisOption("[Sampler] Sigma Churn", float, apply_field("s_churn")),
    AxisOption("[Sampler] Sigma noise", float, apply_field("s_noise")),
    AxisOption("[Sampler] ETA", float, apply_field("eta")),
    AxisOption("[Sampler] Solver order", int, apply_setting("schedulers_solver_order")),
    AxisOption("[Second pass] Upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOption("[Second pass] Sampler", str, apply_hr_sampler_name, fmt=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOption("[Second pass] Denoising Strength", float, apply_field("denoising_strength")),
    AxisOption("[Second pass] Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("[Second pass] CFG scale", float, apply_field("image_cfg_scale")),
    AxisOption("[Second pass] Guidance rescale", float, apply_field("diffusers_guidance_rescale")),
    AxisOption("[Refiner] Model", str, apply_refiner, fmt=format_value, cost=1.0, choices=lambda: ['None'] + sorted(sd_models.checkpoints_list)),
    AxisOption("[Refiner] Refiner start", float, apply_field("refiner_start")),
    AxisOption("[Refiner] Refiner steps", float, apply_field("refiner_steps")),
    AxisOption("[HDR] Clamp boundary", float, apply_field("hdr_boundary")),
    AxisOption("[HDR] Clamp threshold", float, apply_field("hdr_threshold")),
    AxisOption("[HDR] Center channel shift", float, apply_field("hdr_channel_shift")),
    AxisOption("[HDR] Center full shift", float, apply_field("hdr_full_shift")),
    AxisOption("[HDR] Maximize center shift", float, apply_field("hdr_max_center")),
    AxisOption("[HDR] Maximize boundary", float, apply_field("hdr_max_boundry")),
    AxisOption("[ToMe] Token merging ratio (txt2img)", float, apply_override('token_merging_ratio')),
    AxisOption("[ToMe] Token merging ratio (hires)", float, apply_override('token_merging_ratio_hr')),
    AxisOption("[FreeU] 1st stage backbone factor", float, apply_setting('freeu_b1')),
    AxisOption("[FreeU] 2nd stage backbone factor", float, apply_setting('freeu_b2')),
    AxisOption("[FreeU] 1st stage skip factor", float, apply_setting('freeu_s1')),
    AxisOption("[FreeU] 2nd stage skip factor", float, apply_setting('freeu_s2')),
    AxisOption("[IP adapter] Name", str, apply_field('ip_adapter_name'), cost=1.0),
    AxisOption("[IP adapter] Scale", float, apply_field('ip_adapter_scale')),
]


def draw_xyz_grid(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, draw_legend, include_lone_images, include_sub_grids, first_axes_processed, second_axes_processed, margin_size, no_grid):
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    title_texts = [[images.GridAnnotation(z)] for z in z_labels]
    list_size = (len(xs) * len(ys) * len(zs))
    processed_result = None
    shared.state.job_count = list_size * p.n_iter

    def process_cell(x, y, z, ix, iy, iz):
        nonlocal processed_result

        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        shared.state.job = 'grid'
        processed: processing.Processed = cell(x, y, z, ix, iy, iz)
        if processed_result is None:
            processed_result = copy(processed)
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.index_of_first_image = 1
        idx = index(ix, iy, iz)
        if processed is not None and processed.images:
            processed_result.images[idx] = processed.images[0]
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
        else:
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)

    if first_axes_processed == 'x':
        for ix, x in enumerate(xs):
            if second_axes_processed == 'y':
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == 'y':
        for iy, y in enumerate(ys):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == 'z':
        for iz, z in enumerate(zs):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iy, y in enumerate(ys):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)

    if not processed_result:
        shared.log.error("XYZ grid: Failed to initialize processing")
        return processing.Processed(p, [])
    elif not any(processed_result.images):
        shared.log.error("XYZ grid: Failed to return processed image")
        return processing.Processed(p, [])

    z_count = len(zs)
    for i in range(z_count):
        start_index = (i * len(xs) * len(ys)) + i
        end_index = start_index + len(xs) * len(ys)
        if (not no_grid or include_sub_grids) and images.check_grid_size(processed_result.images[start_index:end_index]):
            grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(ys))
            if draw_legend:
                grid = images.draw_grid_annotations(grid, processed_result.images[start_index].size[0], processed_result.images[start_index].size[1], hor_texts, ver_texts, margin_size, title=title_texts[i])
            processed_result.images.insert(i, grid)
        processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
        processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
        processed_result.infotexts.insert(i, processed_result.infotexts[start_index])
    sub_grid_size = processed_result.images[0].size
    if not no_grid and images.check_grid_size(processed_result.images[:z_count]):
        z_grid = images.image_grid(processed_result.images[:z_count], rows=1)
        if draw_legend:
            z_grid = images.draw_grid_annotations(z_grid, sub_grid_size[0], sub_grid_size[1], [[images.GridAnnotation()] for _ in z_labels], [[images.GridAnnotation()]])
        processed_result.images.insert(0, z_grid)
    #processed_result.all_prompts.insert(0, processed_result.all_prompts[0])
    #processed_result.all_seeds.insert(0, processed_result.all_seeds[0])
    processed_result.infotexts.insert(0, processed_result.infotexts[0])
    return processed_result


class SharedSettingsStackHelper(object):
    vae = None
    schedulers_solver_order = None
    token_merging_ratio_hr = None
    token_merging_ratio = None
    sd_model_checkpoint = None
    sd_model_dict = None
    sd_vae_checkpoint = None

    def __enter__(self):
        #Save overridden settings so they can be restored later.
        self.vae = shared.opts.sd_vae
        self.schedulers_solver_order = shared.opts.schedulers_solver_order
        self.token_merging_ratio_hr = shared.opts.token_merging_ratio_hr
        self.token_merging_ratio = shared.opts.token_merging_ratio
        self.sd_model_checkpoint = shared.opts.sd_model_checkpoint
        self.sd_model_dict = shared.opts.sd_model_dict
        self.sd_vae_checkpoint = shared.opts.sd_vae

    def __exit__(self, exc_type, exc_value, tb):
        #Restore overriden settings after plot generation.
        shared.opts.data["sd_vae"] = self.vae
        shared.opts.data["schedulers_solver_order"] = self.schedulers_solver_order
        shared.opts.data["token_merging_ratio_hr"] = self.token_merging_ratio_hr
        shared.opts.data["token_merging_ratio"] = self.token_merging_ratio
        if self.sd_model_dict != shared.opts.sd_model_dict:
            shared.opts.data["sd_model_dict"] = self.sd_model_dict
        if self.sd_model_checkpoint != shared.opts.sd_model_checkpoint:
            shared.opts.data["sd_model_checkpoint"] = self.sd_model_checkpoint
            sd_models.reload_model_weights()
        if self.sd_vae_checkpoint != shared.opts.sd_vae:
            shared.opts.data["sd_vae"] = self.sd_vae_checkpoint
            sd_vae.reload_vae_weights()


re_range = re.compile(r'([-+]?[0-9]*\.?[0-9]+)-([-+]?[0-9]*\.?[0-9]+):?([0-9]+)?')

class Script(scripts.Script):
    current_axis_options = []

    def title(self):
        return "X/Y/Z Grid"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]
        with gr.Row():
            with gr.Column():
                with gr.Row(variant='compact'):
                    x_type = gr.Dropdown(label="X type", container=True, choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("x_type"))
                    x_values = gr.Textbox(label="X values", container=True, lines=1, elem_id=self.elem_id("x_values"))
                    x_values_dropdown = gr.Dropdown(label="X values", container=True, visible=False, multiselect=True, interactive=True)
                    fill_x_button = ToolButton(value=symbols.fill, elem_id="xyz_grid_fill_x_tool_button", visible=False)
                with gr.Row(variant='compact'):
                    y_type = gr.Dropdown(label="Y type", container=True, choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                    y_values = gr.Textbox(label="Y values", container=True, lines=1, elem_id=self.elem_id("y_values"))
                    y_values_dropdown = gr.Dropdown(label="Y values", container=True, visible=False, multiselect=True, interactive=True)
                    fill_y_button = ToolButton(value=symbols.fill, elem_id="xyz_grid_fill_y_tool_button", visible=False)
                with gr.Row(variant='compact'):
                    z_type = gr.Dropdown(label="Z type", container=True, choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("z_type"))
                    z_values = gr.Textbox(label="Z values", container=True, lines=1, elem_id=self.elem_id("z_values"))
                    z_values_dropdown = gr.Dropdown(label="Z values", container=True, visible=False, multiselect=True, interactive=True)
                    fill_z_button = ToolButton(value=symbols.fill, elem_id="xyz_grid_fill_z_tool_button", visible=False)
        with gr.Row():
            with gr.Column():
                csv_mode = gr.Checkbox(label='Text inputs', value=False, elem_id=self.elem_id("csv_mode"), container=False)
                draw_legend = gr.Checkbox(label='Legend', value=True, elem_id=self.elem_id("draw_legend"), container=False)
                no_fixed_seeds = gr.Checkbox(label='Random seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"), container=False)
            with gr.Column():
                no_grid = gr.Checkbox(label='Skip grid', value=False, elem_id=self.elem_id("no_xyz_grid"), container=False)
                include_lone_images = gr.Checkbox(label='Sub-images', value=False, elem_id=self.elem_id("include_lone_images"), container=False)
                include_sub_grids = gr.Checkbox(label='Sub-grids', value=False, elem_id=self.elem_id("include_sub_grids"), container=False)
        with gr.Row():
            margin_size = gr.Slider(label="Grid margins", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))
        with gr.Row():
            swap_xy_axes_button = gr.Button(value="Swap X/Y", elem_id="xy_grid_swap_axes_button", variant="secondary")
            swap_yz_axes_button = gr.Button(value="Swap Y/Z", elem_id="yz_grid_swap_axes_button", variant="secondary")
            swap_xz_axes_button = gr.Button(value="Swap X/Z", elem_id="xz_grid_swap_axes_button", variant="secondary")

        def swap_axes(axis1_type, axis1_values, axis1_values_dropdown, axis2_type, axis2_values, axis2_values_dropdown):
            return self.current_axis_options[axis2_type].label, axis2_values, axis2_values_dropdown, self.current_axis_options[axis1_type].label, axis1_values, axis1_values_dropdown

        xy_swap_args = [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown]
        swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
        yz_swap_args = [y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
        xz_swap_args = [x_type, x_values, x_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

        def fill(axis_type, csv_mode):
            axis = self.current_axis_options[axis_type]
            if axis.choices:
                if csv_mode:
                    return list_to_csv_string(axis.choices()), gr.update()
                else:
                    return gr.update(), axis.choices()
            else:
                return gr.update(), gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values, x_values_dropdown])
        fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values, y_values_dropdown])
        fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values, z_values_dropdown])

        def select_axis(axis_type, axis_values, axis_values_dropdown, csv_mode):
            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None
            current_values = axis_values
            current_dropdown_values = axis_values_dropdown
            if has_choices:
                choices = choices()
                if csv_mode:
                    current_dropdown_values = list(filter(lambda x: x in choices, current_dropdown_values))
                    current_values = list_to_csv_string(current_dropdown_values)
                else:
                    current_dropdown_values = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(axis_values)))]
                    current_dropdown_values = list(filter(lambda x: x in choices, current_dropdown_values))

            return (gr.Button.update(visible=has_choices), gr.Textbox.update(visible=not has_choices or csv_mode, value=current_values),
                    gr.update(choices=choices if has_choices else None, visible=has_choices and not csv_mode, value=current_dropdown_values))

        x_type.change(fn=select_axis, inputs=[x_type, x_values, x_values_dropdown, csv_mode], outputs=[fill_x_button, x_values, x_values_dropdown])
        y_type.change(fn=select_axis, inputs=[y_type, y_values, y_values_dropdown, csv_mode], outputs=[fill_y_button, y_values, y_values_dropdown])
        z_type.change(fn=select_axis, inputs=[z_type, z_values, z_values_dropdown, csv_mode], outputs=[fill_z_button, z_values, z_values_dropdown])

        def change_choice_mode(csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown):
            _fill_x_button, _x_values, _x_values_dropdown = select_axis(x_type, x_values, x_values_dropdown, csv_mode)
            _fill_y_button, _y_values, _y_values_dropdown = select_axis(y_type, y_values, y_values_dropdown, csv_mode)
            _fill_z_button, _z_values, _z_values_dropdown = select_axis(z_type, z_values, z_values_dropdown, csv_mode)
            return _fill_x_button, _x_values, _x_values_dropdown, _fill_y_button, _y_values, _y_values_dropdown, _fill_z_button, _z_values, _z_values_dropdown

        csv_mode.change(fn=change_choice_mode, inputs=[csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown], outputs=[fill_x_button, x_values, x_values_dropdown, fill_y_button, y_values, y_values_dropdown, fill_z_button, z_values, z_values_dropdown])

        def get_dropdown_update_from_params(axis,params):
            val_key = f"{axis} Values"
            vals = params.get(val_key,"")
            valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals))) if x]
            return gr.update(value = valslist)

        self.infotext_fields = (
            (x_type, "X Type"),
            (x_values, "X Values"),
            (x_values_dropdown, lambda params:get_dropdown_update_from_params("X",params)),
            (y_type, "Y Type"),
            (y_values, "Y Values"),
            (y_values_dropdown, lambda params:get_dropdown_update_from_params("Y",params)),
            (z_type, "Z Type"),
            (z_values, "Z Values"),
            (z_values_dropdown, lambda params:get_dropdown_update_from_params("Z",params)),
        )

        return [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, csv_mode, draw_legend, no_fixed_seeds, no_grid, include_lone_images, include_sub_grids, margin_size]

    def run(self, p, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, csv_mode, draw_legend, no_fixed_seeds, no_grid, include_lone_images, include_sub_grids, margin_size): # pylint: disable=W0221
        shared.log.debug(f'xyzgrid: x_type={x_type}|x_values={x_values}|x_values_dropdown={x_values_dropdown}|y_type={y_type}|{y_values}={y_values}|{y_values_dropdown}={y_values_dropdown}|z_type={z_type}|z_values={z_values}|z_values_dropdown={z_values_dropdown}|draw_legend={draw_legend}|include_lone_images={include_lone_images}|include_sub_grids={include_sub_grids}|no_grid={no_grid}|margin_size={margin_size}')
        if not no_fixed_seeds:
            processing.fix_seed(p)
        if not shared.opts.return_grid:
            p.batch_size = 1
        def process_axis(opt, vals, vals_dropdown):
            if opt.label == 'Nothing':
                return [0]
            if opt.choices is not None and not csv_mode:
                valslist = vals_dropdown
            else:
                valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals))) if x]
            if opt.type == int:
                valslist_ext = []
                for val in valslist:
                    m = re_range.fullmatch(val)
                    if m is not None:
                        start_val = int(m.group(1)) if m.group(1) is not None else val
                        end_val = int(m.group(2)) if m.group(2) is not None else val
                        num = int(m.group(3)) if m.group(3) is not None else int(end_val-start_val)
                        valslist_ext += [int(x) for x in np.linspace(start=start_val, stop=end_val, num=max(2, num)).tolist()]
                        shared.log.debug(f'XYZ grid range: start={start_val} end={end_val} num={max(2, num)} list={valslist}')
                    else:
                        valslist_ext.append(int(val))
                valslist.clear()
                valslist = [x for x in valslist_ext if x not in valslist]
            elif opt.type == float:
                valslist_ext = []
                for val in valslist:
                    m = re_range.fullmatch(val)
                    if m is not None:
                        start_val = float(m.group(1)) if m.group(1) is not None else val
                        end_val = float(m.group(2)) if m.group(2) is not None else val
                        num = int(m.group(3)) if m.group(3) is not None else int(end_val-start_val)
                        valslist_ext += [round(float(x), 2) for x in np.linspace(start=start_val, stop=end_val, num=max(2, num)).tolist()]
                        shared.log.debug(f'XYZ grid range: start={start_val} end={end_val} num={max(2, num)} list={valslist}')
                    else:
                        valslist_ext.append(float(val))
                valslist.clear()
                valslist = [x for x in valslist_ext if x not in valslist]
            elif opt.type == str_permutations: # pylint: disable=comparison-with-callable
                valslist = list(permutations(valslist))
            valslist = [opt.type(x) for x in valslist]
            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)
            return valslist

        x_opt = self.current_axis_options[x_type]
        if x_opt.choices is not None and not csv_mode:
            x_values = list_to_csv_string(x_values_dropdown)
        xs = process_axis(x_opt, x_values, x_values_dropdown)
        y_opt = self.current_axis_options[y_type]
        if y_opt.choices is not None and not csv_mode:
            y_values = list_to_csv_string(y_values_dropdown)
        ys = process_axis(y_opt, y_values, y_values_dropdown)
        z_opt = self.current_axis_options[z_type]
        if z_opt.choices is not None and not csv_mode:
            z_values = list_to_csv_string(z_values_dropdown)
        zs = process_axis(z_opt, z_values, z_values_dropdown)
        Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['Seed', 'Var. seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)
            zs = fix_axis_seeds(z_opt, zs)

        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys) * len(zs)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs) * len(zs)
        elif z_opt.label == 'Steps':
            total_steps = sum(zs) * len(xs) * len(ys)
        else:
            total_steps = p.steps * len(xs) * len(ys) * len(zs)
        if isinstance(p, processing.StableDiffusionProcessingTxt2Img) and p.enable_hr:
            if x_opt.label == "Hires steps":
                total_steps += sum(xs) * len(ys) * len(zs)
            elif y_opt.label == "Hires steps":
                total_steps += sum(ys) * len(xs) * len(zs)
            elif z_opt.label == "Hires steps":
                total_steps += sum(zs) * len(xs) * len(ys)
            elif p.hr_second_pass_steps:
                total_steps += p.hr_second_pass_steps * len(xs) * len(ys) * len(zs)
            else:
                total_steps *= 2
        total_steps *= p.n_iter
        image_cell_count = p.n_iter * p.batch_size
        shared.log.info(f"XYZ grid: images={len(xs)*len(ys)*len(zs)*image_cell_count} grid={len(zs)} {len(xs)}x{len(ys)} cells={len(zs)} steps={total_steps}")
        AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])
        shared.state.xyz_plot_x = AxisInfo(x_opt, xs)
        shared.state.xyz_plot_y = AxisInfo(y_opt, ys)
        shared.state.xyz_plot_z = AxisInfo(z_opt, zs)
        # If one of the axes is very slow to change between (like SD model checkpoint), then make sure it is in the outer iteration of the nested `for` loop.
        first_axes_processed = 'z'
        second_axes_processed = 'y'
        if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
            first_axes_processed = 'x'
            if y_opt.cost > z_opt.cost:
                second_axes_processed = 'y'
            else:
                second_axes_processed = 'z'
        elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
            first_axes_processed = 'y'
            if x_opt.cost > z_opt.cost:
                second_axes_processed = 'x'
            else:
                second_axes_processed = 'z'
        elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
            first_axes_processed = 'z'
            if x_opt.cost > y_opt.cost:
                second_axes_processed = 'x'
            else:
                second_axes_processed = 'y'
        grid_infotext = [None] * (1 + len(zs))

        def cell(x, y, z, ix, iy, iz):
            if shared.state.interrupted:
                return processing.Processed(p, [], p.seed, "")
            pc = copy(p)
            pc.override_settings_restore_afterwards = False
            pc.styles = pc.styles[:]
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)
            z_opt.apply(pc, z, zs)
            try:
                res = processing.process_images(pc)
            except Exception as e:
                shared.log.error(f"XYZ grid: Failed to process image: {e}")
                errors.display(e, 'XYZ grid')
                res = None
            subgrid_index = 1 + iz # Sets subgrid infotexts
            if grid_infotext[subgrid_index] is None and ix == 0 and iy == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                pc.extra_generation_params['Script'] = self.title()
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
                grid_infotext[subgrid_index] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            if grid_infotext[0] is None and ix == 0 and iy == 0 and iz == 0: # Sets main grid infotext
                pc.extra_generation_params = copy(pc.extra_generation_params)
                if z_opt.label != 'Nothing':
                    pc.extra_generation_params["Z Type"] = z_opt.label
                    pc.extra_generation_params["Z Values"] = z_values
                    if z_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs])
                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            return res

        with SharedSettingsStackHelper():
            processed = draw_xyz_grid(
                p,
                xs=xs,
                ys=ys,
                zs=zs,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys],
                z_labels=[z_opt.format_value(p, z_opt, z) for z in zs],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images,
                include_sub_grids=include_sub_grids,
                first_axes_processed=first_axes_processed,
                second_axes_processed=second_axes_processed,
                margin_size=margin_size,
                no_grid=no_grid,
            )

        if not processed.images:
            return processed # It broke, no further handling needed.
        z_count = len(zs)
        processed.infotexts[:1+z_count] = grid_infotext[:1+z_count] # Set the grid infotexts to the real ones with extra_generation_params (1 main grid + z_count sub-grids)
        if not include_lone_images:
             # Don't need sub-images anymore, drop from list:
            if no_grid and include_sub_grids:
                processed.images = processed.images[:z_count] # we don't have the main grid image, and need zero additional sub-images
            else:
                processed.images = processed.images[:z_count+1] # we either have the main grid image, or need one sub-images
        if shared.opts.grid_save: # Auto-save main and sub-grids:
            grid_count = z_count + ( 1 if not no_grid and z_count > 1 else 0 )
            for g in range(grid_count):
                adj_g = g-1 if g > 0 else g
                images.save_image(processed.images[g], p.outpath_grids, "xyz_grid", info=processed.infotexts[g], extension=shared.opts.grid_format, prompt=processed.all_prompts[adj_g], seed=processed.all_seeds[adj_g], grid=True, p=processed)
        if not include_sub_grids: # Done with sub-grids, drop all related information:
            for _sg in range(z_count):
                del processed.images[1]
                del processed.all_prompts[1]
                del processed.all_seeds[1]
                del processed.infotexts[1]
        elif no_grid:
            # del processed.images[0]
            # del processed.all_prompts[0]
            # del processed.all_seeds[0]
            del processed.infotexts[0]
        return processed
