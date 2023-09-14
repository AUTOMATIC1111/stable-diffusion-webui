import os
import re
import html
import json
import shutil

import torch
import tqdm
import gradio as gr
import safetensors.torch

from modules import shared, images, sd_models, sd_vae, sd_models_config


checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]


def run_pnginfo(image):
    if image is None:
        return '', '', ''
    geninfo, items = images.read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}
    info = ''
    for key, text in items.items():
        if key != 'UserComment':
            info += f"<div><b>{html.escape(str(key))}</b>: {html.escape(str(text))}</div>"
    return '', geninfo, info


def create_config(ckpt_result, config_source, a, b, c):
    def config(x):
        res = sd_models_config.find_checkpoint_config_near_filename(x) if x else None
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
    shared.log.info("Copying config: {cfg} -> {checkpoint_filename}")
    shutil.copyfile(cfg, checkpoint_filename)


def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float:
        return tensor.half()
    return tensor


def run_modelmerger(id_task, primary_model_name, secondary_model_name, tertiary_model_name, interp_method, multiplier, save_as_half, custom_name, checkpoint_format, config_source, bake_in_vae, discard_weights, save_metadata): # pylint: disable=unused-argument
    shared.state.begin('model-merge')
    save_as_half = save_as_half == 0

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
    if not primary_model_name or primary_model_name == 'None':
        return fail("Failed: Merging requires a primary model.")
    primary_model_info = sd_models.checkpoints_list[primary_model_name]
    if theta_func2 and (not secondary_model_name or secondary_model_name == 'None'):
        return fail("Failed: Merging requires a secondary model.")
    secondary_model_info = sd_models.checkpoints_list[secondary_model_name] if theta_func2 else None
    if theta_func1 and (not tertiary_model_name or tertiary_model_name == 'None'):
        return fail(f"Failed: Interpolation method ({interp_method}) requires a tertiary model.")
    tertiary_model_info = sd_models.checkpoints_list[tertiary_model_name] if theta_func1 else None
    result_is_inpainting_model = False
    result_is_instruct_pix2pix_model = False
    if theta_func2:
        shared.state.textinfo = "Loading B"
        shared.log.info(f"Model merge loading secondary model: {secondary_model_info.filename}")
        theta_1 = sd_models.read_state_dict(secondary_model_info.filename)
    else:
        theta_1 = None
    if theta_func1:
        shared.state.textinfo = "Loading C"
        shared.log.info(f"Model merge loading tertiary model: {tertiary_model_info.filename}")
        theta_2 = sd_models.read_state_dict(tertiary_model_info.filename)
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
    shared.log.info(f"Model merge loading primary model: {primary_model_info.filename}")
    theta_0 = sd_models.read_state_dict(primary_model_info.filename)
    shared.log.info("Model merge: running")
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
                if a.shape[1] == 4 and b.shape[1] == 8:
                    raise RuntimeError("When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.")
                if a.shape[1] == 8 and b.shape[1] == 4:#If we have an Instruct-Pix2Pix model...
                    theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)#Merge only the vectors the models have in common.  Otherwise we get an error due to dimension mismatch.
                    result_is_instruct_pix2pix_model = True
                else:
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
        shared.log.info(f"Model merge: baking in VAE: {bake_in_vae_filename}")
        shared.state.textinfo = 'Baking in VAE'
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename)
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
    ckpt_dir = shared.opts.ckpt_dir or sd_models.model_path
    filename = filename_generator() if custom_name == '' else custom_name
    filename += ".inpainting" if result_is_inpainting_model else ""
    filename += ".instruct-pix2pix" if result_is_instruct_pix2pix_model else ""
    filename += "." + checkpoint_format
    output_modelname = os.path.join(ckpt_dir, filename)
    shared.state.nextjob()
    shared.state.textinfo = "Saving"
    metadata = None
    if save_metadata:
        metadata = {"format": "pt", "sd_merge_models": {}}
        merge_recipe = {
            "type": "webui", # indicate this model was merged with webui's built-in merger
            "primary_model_hash": primary_model_info.sha256,
            "secondary_model_hash": secondary_model_info.sha256 if secondary_model_info else None,
            "tertiary_model_hash": tertiary_model_info.sha256 if tertiary_model_info else None,
            "interp_method": interp_method,
            "multiplier": multiplier,
            "save_as_half": save_as_half,
            "custom_name": custom_name,
            "config_source": config_source,
            "bake_in_vae": bake_in_vae,
            "discard_weights": discard_weights,
            "is_inpainting": result_is_inpainting_model,
            "is_instruct_pix2pix": result_is_instruct_pix2pix_model
        }
        metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

        def add_model_metadata(checkpoint_info):
            checkpoint_info.calculate_shorthash()
            metadata["sd_merge_models"][checkpoint_info.sha256] = {
                "name": checkpoint_info.name,
                "legacy_hash": checkpoint_info.hash,
                "sd_merge_recipe": checkpoint_info.metadata.get("sd_merge_recipe", None)
            }
            metadata["sd_merge_models"].update(checkpoint_info.metadata.get("sd_merge_models", {}))

        add_model_metadata(primary_model_info)
        if secondary_model_info:
            add_model_metadata(secondary_model_info)
        if tertiary_model_info:
            add_model_metadata(tertiary_model_info)
        metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

    _, extension = os.path.splitext(output_modelname)
    if extension.lower() == ".safetensors":
        safetensors.torch.save_file(theta_0, output_modelname, metadata=metadata)
    else:
        torch.save(theta_0, output_modelname)
    sd_models.list_models()
    created_model = next((ckpt for ckpt in sd_models.checkpoints_list.values() if ckpt.name == filename), None)
    if created_model:
        created_model.calculate_shorthash()
    create_config(output_modelname, config_source, primary_model_info, secondary_model_info, tertiary_model_info)
    shared.log.info(f"Model merge saved: {output_modelname}.")
    shared.state.textinfo = "Checkpoint saved"
    shared.state.end()
    return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], "Checkpoint saved to " + output_modelname]

def run_modelconvert(model, checkpoint_formats, precision, conv_type, custom_name, unet_conv, text_encoder_conv, vae_conv, others_conv, fix_clip):

    # position_ids in clip is int64. model_ema.num_updates is int32
    dtypes_to_fp16 = {torch.float32, torch.float64, torch.bfloat16}
    dtypes_to_bf16 = {torch.float32, torch.float64, torch.float16}

    def conv_fp16(t: torch.Tensor):
        return t.half() if t.dtype in dtypes_to_fp16 else t

    def conv_bf16(t: torch.Tensor):
        return t.bfloat16() if t.dtype in dtypes_to_bf16 else t

    def conv_full(t):
        return t

    _g_precision_func = {
        "full": conv_full,
        "fp32": conv_full,
        "fp16": conv_fp16,
        "bf16": conv_bf16,
    }

    def check_weight_type(k: str) -> str:
        if k.startswith("model.diffusion_model"):
            return "unet"
        elif k.startswith("first_stage_model"):
            return "vae"
        elif k.startswith("cond_stage_model"):
            return "clip"
        return "other"

    def load_model(path):
        if path.endswith(".safetensors"):
            m = safetensors.torch.load_file(path, device="cpu")
        else:
            m = torch.load(path, map_location="cpu")
        state_dict = m["state_dict"] if "state_dict" in m else m
        return state_dict


    def fix_model(model, fix_clip=False):
        # code from model-toolkit
        nai_keys = {
            'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
            'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
            'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.'
        }
        for k in list(model.keys()):
            for r in nai_keys:
                if type(k) == str and k.startswith(r):
                    new_key = k.replace(r, nai_keys[r])
                    model[new_key] = model[k]
                    del model[k]
                    shared.log.warning(f"Model convert: fixed NovelAI error key: {k}")
                    break
        if fix_clip:
            i = "cond_stage_model.transformer.text_model.embeddings.position_ids"
            if i in model:
                correct = torch.Tensor([list(range(77))]).to(torch.int64)
                now = model[i].to(torch.int64)

                broken = correct.ne(now)
                broken = [i for i in range(77) if broken[0][i]]
                model[i] = correct
                if len(broken) != 0:
                    shared.log.warning(f"Model convert: fixed broken CLiP: {broken}")

        return model

    if model == "":
        return "Error: you must choose a model"
    if len(checkpoint_formats) == 0:
        return "Error: at least choose one model save format"

    extra_opt = {
        "unet": unet_conv,
        "clip": text_encoder_conv,
        "vae": vae_conv,
        "other": others_conv
    }
    shared.state.begin('model-convert')
    model_info = sd_models.checkpoints_list[model]
    shared.state.textinfo = f"Loading {model_info.filename}..."
    shared.log.info(f"Model convert loading: {model_info.filename}")
    state_dict = load_model(model_info.filename)

    ok = {}  # {"state_dict": {}}

    conv_func = _g_precision_func[precision]

    def _hf(wk: str, t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            return
        w_t = check_weight_type(wk)
        conv_t = extra_opt[w_t]
        if conv_t == "convert":
            ok[wk] = conv_func(t)
        elif conv_t == "copy":
            ok[wk] = t
        elif conv_t == "delete":
            return
    shared.log.info("Model convert: running")
    if conv_type == "ema-only":
        for k in tqdm.tqdm(state_dict):
            ema_k = "___"
            try:
                ema_k = "model_ema." + k[6:].replace(".", "")
            except Exception:
                pass
            if ema_k in state_dict:
                _hf(k, state_dict[ema_k])
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                _hf(k, state_dict[k])
    elif conv_type == "no-ema":
        for k, v in tqdm.tqdm(state_dict.items()):
            if "model_ema." not in k:
                _hf(k, v)
    else:
        for k, v in tqdm.tqdm(state_dict.items()):
            _hf(k, v)

    ok = fix_model(ok, fix_clip=fix_clip)
    output = ""
    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
    save_name = f"{model_info.model_name}-{precision}"
    if conv_type != "disabled":
        save_name += f"-{conv_type}"
    if custom_name != "":
        save_name = custom_name
    for fmt in checkpoint_formats:
        ext = ".safetensors" if fmt == "safetensors" else ".ckpt"
        _save_name = save_name + ext
        save_path = os.path.join(ckpt_dir, _save_name)
        shared.log.info(f"Model convert saving: {save_path}")
        if fmt == "safetensors":
            safetensors.torch.save_file(ok, save_path)
        else:
            torch.save({"state_dict": ok}, save_path)
        output += f"Checkpoint saved to {save_path}<br>"
    shared.state.end()
    return output
