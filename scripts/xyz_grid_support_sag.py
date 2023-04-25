"""
MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import os.path
import modules.scripts as scripts


def update_script_args(p, value, arg_idx):
    for s in scripts.scripts_txt2img.alwayson_scripts:
        args = list(p.script_args)
        # print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx - 1]} to {value}")
        args[s.args_from + arg_idx] = value
        p.script_args = tuple(args)
        break


def apply_module(p, x, xs, i):
    update_script_args(p, True, 0)  # set Enabled to True
    update_script_args(p, x, 2 + 4 * i)  # enabled, separate_weights, ({module}, model, weight_unet, weight_tenc), ...


def apply_weight(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 4 + 4 * i)  # enabled, separate_weights, (module, model, {weight_unet, weight_tenc}), ...
    update_script_args(p, x, 5 + 4 * i)


def apply_weight_unet(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 4 + 4 * i)  # enabled, separate_weights, (module, model, {weight_unet}, weight_tenc), ...


def apply_weight_tenc(p, x, xs, i):
    update_script_args(p, True, 0)
    update_script_args(p, x, 5 + 4 * i)  # enabled, separate_weights, (module, model, weight_unet, {weight_tenc}), ...


def apply_sag_guidance_scale(p, x, xs):
    update_script_args(p, x, 0)
    update_script_args(p, x, 1)  # sag_guidance_scale


def apply_sag_mask_threshold(p, x, xs):
    update_script_args(p, x, 0)
    update_script_args(p, x, 2)  # sag_mask_threshold


for scriptDataTuple in scripts.scripts_data:
    if os.path.basename(scriptDataTuple.path) == "xy_grid.py" or os.path.basename(
            scriptDataTuple.path) == "xyz_grid.py":
        xy_grid = scriptDataTuple.module
        sag_guidance_scale = xy_grid.AxisOption("SAG Guidance Scale", float,
                                                lambda p, x, xs: apply_sag_guidance_scale(p, x, xs),
                                                xy_grid.format_value_add_label, None, cost=0.5)
        sag_mask_threshold = xy_grid.AxisOption("SAG Mask Threshold", float,
                                                lambda p, x, xs: apply_sag_mask_threshold(p, x, xs),
                                                xy_grid.format_value_add_label, None, cost=0.5)
        xy_grid.axis_options.extend([sag_guidance_scale, sag_mask_threshold])
