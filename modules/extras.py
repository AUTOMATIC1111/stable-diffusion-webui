import os
import re
import shutil


import torch
import tqdm

from modules import shared, images, sd_models, sd_vae
from modules.ui_common import plaintext_to_html
import gradio as gr
import safetensors.torch


def run_pnginfo(image):
    if image is None:
        return '', '', ''

    geninfo, items = images.read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}

    info = ''
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', geninfo, info


def create_config(ckpt_result, config_source, a, b, c):
    def config(x):
        res = sd_models.find_checkpoint_config(x) if x else None
        return res if res != shared.sd_default_config else None

    if config_source == 0:
        cfg = config(a) or config(b) or config(c)
    elif config_source == 1:
        cfg = config(b)
    elif config_source == 2:
        cfg = config(c)
    else:
        cfg = None

    if cfg is None:
        return

    filename, _ = os.path.splitext(ckpt_result)
    checkpoint_filename = filename + ".yaml"

    print("Copying config:")
    print("   from:", cfg)
    print("     to:", checkpoint_filename)
    shutil.copyfile(cfg, checkpoint_filename)


checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]


def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float:
        return tensor.half()

    return tensor


def run_modelmerger(id_task, primary_model_name, secondary_model_name, tertiary_model_name, interp_method, multiplier, save_as_half, custom_name, checkpoint_format, config_source, bake_in_vae, discard_weights):
    shared.state.begin()
    shared.state.job = 'model-merge'

    def fail(message):
        shared.state.textinfo = message
        shared.state.end()
        return [*[gr.update() for _ in range(4)], message]

    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    def filename_weighted_sum():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        Ma = round(1 - multiplier, 2)
        Mb = round(multiplier, 2)

        return f"{Ma}({a}) + {Mb}({b})"

    def filename_add_difference():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        c = tertiary_model_info.model_name
        M = round(multiplier, 2)

        return f"{a} + {M}({b} - {c})"

    def filename_nothing():
        return primary_model_info.model_name

    theta_funcs = {
        "Weighted sum": (filename_weighted_sum, None, weighted_sum),
        "Add difference": (filename_add_difference, get_difference, add_difference),
        "No interpolation": (filename_nothing, None, None),
    }
    filename_generator, theta_func1, theta_func2 = theta_funcs[interp_method]
    shared.state.job_count = (1 if theta_func1 else 0) + (1 if theta_func2 else 0)

    if not primary_model_name:
        return fail("Failed: Merging requires a primary model.")

    primary_model_info = sd_models.checkpoints_list[primary_model_name]

    if theta_func2 and not secondary_model_name:
        return fail("Failed: Merging requires a secondary model.")

    secondary_model_info = sd_models.checkpoints_list[secondary_model_name] if theta_func2 else None

    if theta_func1 and not tertiary_model_name:
        return fail(f"Failed: Interpolation method ({interp_method}) requires a tertiary model.")

    tertiary_model_info = sd_models.checkpoints_list[tertiary_model_name] if theta_func1 else None

    result_is_inpainting_model = False

    if theta_func2:
        shared.state.textinfo = f"Loading B"
        print(f"Loading {secondary_model_info.filename}...")
        theta_1 = sd_models.read_state_dict(secondary_model_info.filename, map_location='cpu')
    else:
        theta_1 = None

    if theta_func1:
        shared.state.textinfo = f"Loading C"
        print(f"Loading {tertiary_model_info.filename}...")
        theta_2 = sd_models.read_state_dict(tertiary_model_info.filename, map_location='cpu')

        shared.state.textinfo = 'Merging B and C'
        shared.state.sampling_steps = len(theta_1.keys())
        for key in tqdm.tqdm(theta_1.keys()):
            if key in checkpoint_dict_skip_on_merge:
                continue

            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])

            shared.state.sampling_step += 1
        del theta_2

        shared.state.nextjob()

    shared.state.textinfo = f"Loading {primary_model_info.filename}..."
    print(f"Loading {primary_model_info.filename}...")
    theta_0 = sd_models.read_state_dict(primary_model_info.filename, map_location='cpu')

    print("Merging...")
    shared.state.textinfo = 'Merging A and B'
    shared.state.sampling_steps = len(theta_0.keys())
    for key in tqdm.tqdm(theta_0.keys()):
        if theta_1 and 'model' in key and key in theta_1:

            if key in checkpoint_dict_skip_on_merge:
                continue

            a = theta_0[key]
            b = theta_1[key]

            # this enables merging an inpainting model (A) with another one (B);
            # where normal model would have 4 channels, for latenst space, inpainting model would
            # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
            if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                if a.shape[1] == 4 and b.shape[1] == 9:
                    raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")

                assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"

                theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                result_is_inpainting_model = True
            else:
                theta_0[key] = theta_func2(a, b, multiplier)

            theta_0[key] = to_half(theta_0[key], save_as_half)

        shared.state.sampling_step += 1

    del theta_1

    bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
    if bake_in_vae_filename is not None:
        print(f"Baking in VAE from {bake_in_vae_filename}")
        shared.state.textinfo = 'Baking in VAE'
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location='cpu')

        for key in vae_dict.keys():
            theta_0_key = 'first_stage_model.' + key
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = to_half(vae_dict[key], save_as_half)

        del vae_dict

    if save_as_half and not theta_func2:
        for key in theta_0.keys():
            theta_0[key] = to_half(theta_0[key], save_as_half)

    if discard_weights:
        regex = re.compile(discard_weights)
        for key in list(theta_0):
            if re.search(regex, key):
                theta_0.pop(key, None)

    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

    filename = filename_generator() if custom_name == '' else custom_name
    filename += ".inpainting" if result_is_inpainting_model else ""
    filename += "." + checkpoint_format

    output_modelname = os.path.join(ckpt_dir, filename)

    shared.state.nextjob()
    shared.state.textinfo = "Saving"
    print(f"Saving to {output_modelname}...")

    _, extension = os.path.splitext(output_modelname)
    if extension.lower() == ".safetensors":
        safetensors.torch.save_file(theta_0, output_modelname, metadata={"format": "pt"})
    else:
        torch.save(theta_0, output_modelname)

    sd_models.list_models()

    create_config(output_modelname, config_source, primary_model_info, secondary_model_info, tertiary_model_info)

    print(f"Checkpoint saved to {output_modelname}.")
    shared.state.textinfo = "Checkpoint saved"
    shared.state.end()

    return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], "Checkpoint saved to " + output_modelname]
