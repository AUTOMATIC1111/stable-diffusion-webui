from collections import namedtuple
from copy import copy
import random

import modules.scripts as scripts
import gradio as gr

from modules.hypernetworks import hypernetwork
from modules.processing import process_images, Processed, StableDiffusionProcessing, StableDiffusionProcessingTxt2Img
from modules.shared import opts
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models


def apply_field(field):
    def fun(p, x):
        setattr(p, field, x)

    return fun


def apply_hypernetwork_strength(p, x):
    hypernetwork.apply_strength(x)


def apply_clip_skip(p, x):
    opts.data["CLIP_stop_at_last_layers"] = x


AxisOption = namedtuple("AxisOption", ["label", "type", "apply"])

axis_options = [
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOption("Hypernet str.", float, apply_hypernetwork_strength),
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    AxisOption("Eta", float, apply_field("eta")),
    AxisOption("Clip skip", int, apply_clip_skip),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
]

class Script(scripts.Script):
    def title(self):
        return "Random Range"

    def ui(self, is_img2img):
        control_list = []
        for opt_idx, opt in enumerate(axis_options):
            with gr.Row():
                control_list.append(gr.Textbox(opt.label, interactive=False, visible=False, lines=1))
                control_list.append(gr.Textbox(label="Min Value", visible=False, lines=1))
                control_list.append(gr.Textbox(label="Max Value", visible=False, lines=1))

        return control_list

    def run(self, p, *control_list):
        p:StableDiffusionProcessing = copy(p)
        processed_result = None

        CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        batch_count = p.n_iter
        p.n_iter = 1
        for n in range(batch_count):
            applied_settings = []
            for opt_idx, (ui_value_min, ui_value_max) in enumerate(zip(control_list[1::3], control_list[2::3])):
                if ui_value_min == "" and ui_value_max == "":
                    continue
                axis_option = axis_options[opt_idx]
                if axis_option.type is float:
                    val = random.uniform(float(ui_value_min), float(ui_value_max))
                    val = round(val, 2)
                elif axis_option.type is int:
                    val = random.randint(int(ui_value_min), int(ui_value_max))
                else:
                    raise Exception(f"random_range: Unknown type: {axis_option.type}")

                axis_option.apply(p, val)
                applied_settings.append((axis_option.label, val))

            if len(applied_settings) == 0:
                raise Exception("random_range: No settings specified.")

            settings_str = ", ".join([f"{label}: {val}" for label,val in applied_settings])
            print(f"random_range overriding {len(applied_settings)} settings: {settings_str}")

            if batch_count > 1:
                shared.state.job = f"Batch {n+1} out of {batch_count}"
            processed:Processed = process_images(p)
            if not processed_result:
                processed_result = processed
            else:
                try:
                    processed_result.images.append(processed.images[0])
                    processed_result.all_prompts.append(processed.prompt)
                    processed_result.all_seeds.append(processed.seed)
                    processed_result.infotexts.append(processed.infotexts[0])
                except: pass

        # restore checkpoint in case it was changed by axes
        modules.sd_models.reload_model_weights(shared.sd_model)

        hypernetwork.load_hypernetwork(opts.sd_hypernetwork)
        hypernetwork.apply_strength()

        opts.data["CLIP_stop_at_last_layers"] = CLIP_stop_at_last_layers

        return processed_result
