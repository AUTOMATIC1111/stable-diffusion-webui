import os
import html
import json
import time
import shutil

import torch
import tqdm
import gradio as gr
import safetensors.torch
from modules.merging.merge import merge_models
from modules.merging.merge_utils import TRIPLE_METHODS

from modules import shared, images, sd_models, sd_vae, sd_models_config, devices


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


def run_modelmerger(id_task, **kwargs):  # pylint: disable=unused-argument
    shared.state.begin('merge')
    t0 = time.time()

    def fail(message):
        shared.state.textinfo = message
        shared.state.end()
        return [*[gr.update() for _ in range(4)], message]

    kwargs["models"] = {
        "model_a": sd_models.get_closet_checkpoint_match(kwargs.get("primary_model_name", None)).filename,
        "model_b": sd_models.get_closet_checkpoint_match(kwargs.get("secondary_model_name", None)).filename,
    }

    if kwargs.get("primary_model_name", None) in [None, 'None']:
        return fail("Failed: Merging requires a primary model.")
    primary_model_info = sd_models.get_closet_checkpoint_match(kwargs.get("primary_model_name", None))
    if kwargs.get("secondary_model_name", None) in [None, 'None']:
        return fail("Failed: Merging requires a secondary model.")
    secondary_model_info = sd_models.get_closet_checkpoint_match(kwargs.get("secondary_model_name", None))
    if kwargs.get("tertiary_model_name", None) in [None, 'None'] and kwargs.get("merge_mode", None) in TRIPLE_METHODS:
        return fail(f"Failed: Interpolation method ({kwargs.get('merge_mode', None)}) requires a tertiary model.")
    tertiary_model_info = sd_models.get_closet_checkpoint_match(kwargs.get("tertiary_model_name", None)) if kwargs.get("merge_mode", None) in TRIPLE_METHODS else None

    del kwargs["primary_model_name"]
    del kwargs["secondary_model_name"]
    if kwargs.get("tertiary_model_name", None) is not None:
        kwargs["models"] |= {"model_c": sd_models.get_closet_checkpoint_match(kwargs.get("tertiary_model_name", None)).filename}
        del kwargs["tertiary_model_name"]

    if hasattr(kwargs, "alpha_base") and hasattr(kwargs, "alpha_in_blocks") and hasattr(kwargs, "alpha_mid_block") and hasattr(kwargs, "alpha_out_blocks"):
        try:
            alpha = [float(x) for x in
                    [kwargs["alpha_base"]] + kwargs["alpha_in_blocks"].split(",") + [kwargs["alpha_mid_block"]] + kwargs["alpha_out_blocks"].split(",")]
            assert len(alpha) == 26 or len(alpha) == 20, "Alpha Block Weights are wrong length (26 or 20 for SDXL) falling back"
            kwargs["alpha"] = alpha
        except KeyError as ke:
            shared.log.warning(f"Merge: Malformed manual block weight: {ke}")
    elif hasattr(kwargs, "alpha_preset") or hasattr(kwargs, "alpha"):
        kwargs["alpha"] = kwargs.get("alpha_preset", kwargs["alpha"])

    kwargs.pop("alpha_base", None)
    kwargs.pop("alpha_in_blocks", None)
    kwargs.pop("alpha_mid_block", None)
    kwargs.pop("alpha_out_blocks", None)
    kwargs.pop("alpha_preset", None)

    if hasattr(kwargs, "beta_base") and hasattr(kwargs, "beta_in_blocks") and hasattr(kwargs, "beta_mid_block") and hasattr(kwargs, "beta_out_blocks"):
        try:
            beta = [float(x) for x in
                    [kwargs["beta_base"]] + kwargs["beta_in_blocks"].split(",") + [kwargs["beta_mid_block"]] + kwargs["beta_out_blocks"].split(",")]
            assert len(beta) == 26 or len(beta) == 20, "Beta Block Weights are wrong length (26 or 20 for SDXL) falling back"
            kwargs["beta"] = beta
        except KeyError as ke:
            shared.log.warning(f"Merge: Malformed manual block weight: {ke}")
    elif hasattr(kwargs, "beta_preset") or hasattr(kwargs, "beta"):
        kwargs["beta"] = kwargs.get("beta_preset", kwargs["beta"])

    kwargs.pop("beta_base", None)
    kwargs.pop("beta_in_blocks", None)
    kwargs.pop("beta_mid_block", None)
    kwargs.pop("beta_out_blocks", None)
    kwargs.pop("beta_preset", None)

    if kwargs["device"] == "gpu":
        kwargs["device"] = devices.device
    elif kwargs["device"] == "shuffle":
        kwargs["device"] = torch.device("cpu")
        kwargs["work_device"] = devices.device
    else:
        kwargs["device"] = torch.device("cpu")
    if kwargs.pop("unload", False):
        sd_models.unload_model_weights()

    try:
        theta_0 = merge_models(**kwargs)
    except Exception as e:
        return fail(f"{e}")

    try:
        theta_0 = theta_0.to_dict() #TensorDict -> Dict if necessary
    except Exception:
        pass

    bake_in_vae_filename = sd_vae.vae_dict.get(kwargs.get("bake_in_vae", None), None)
    if bake_in_vae_filename is not None:
        shared.log.info(f"Merge VAE='{bake_in_vae_filename}'")
        shared.state.textinfo = 'Merge VAE'
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename)
        for key in vae_dict.keys():
            theta_0_key = 'first_stage_model.' + key
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = to_half(vae_dict[key], kwargs.get("precision", "fp16") == "fp16")
        del vae_dict

    ckpt_dir = shared.opts.ckpt_dir or sd_models.model_path
    filename = kwargs.get("custom_name", "Unnamed_Merge")
    filename += "." + kwargs.get("checkpoint_format", None)
    output_modelname = os.path.join(ckpt_dir, filename)
    shared.state.textinfo = "merge saving"
    metadata = None
    if kwargs.get("save_metadata", False):
        metadata = {"format": "pt", "sd_merge_models": {}}
        merge_recipe = {
            "type": "SDNext",  # indicate this model was merged with webui's built-in merger
            "primary_model_hash": primary_model_info.sha256,
            "secondary_model_hash": secondary_model_info.sha256 if secondary_model_info else None,
            "tertiary_model_hash": tertiary_model_info.sha256 if tertiary_model_info else None,
            "merge_mode": kwargs.get('merge_mode', None),
            "alpha": kwargs.get('alpha', None),
            "beta": kwargs.get('beta', None),
            "precision": kwargs.get('precision', None),
            "custom_name": kwargs.get("custom_name", "Unamed_Merge"),
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

    if os.path.exists(output_modelname) and not kwargs.get("overwrite", False):
        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Model alredy exists: {output_modelname}"]
    if extension.lower() == ".safetensors":
        safetensors.torch.save_file(theta_0, output_modelname, metadata=metadata)
    else:
        torch.save(theta_0, output_modelname)

    t1 = time.time()
    shared.log.info(f"Merge complete: saved='{output_modelname}' time={t1-t0:.2f}")
    sd_models.list_models()
    created_model = next((ckpt for ckpt in sd_models.checkpoints_list.values() if ckpt.name == filename), None)
    if created_model:
        created_model.calculate_shorthash()
    devices.torch_gc(force=True)
    shared.state.end()
    return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Model saved to {output_modelname}"]


def run_modelconvert(model, checkpoint_formats, precision, conv_type, custom_name, unet_conv, text_encoder_conv,
                     vae_conv, others_conv, fix_clip):
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
    shared.state.begin('convert')
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
