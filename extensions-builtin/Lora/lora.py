import glob
import os
import re
import torch
from typing import Union

from modules import shared, devices, sd_models, errors

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}

        _, ext = os.path.splitext(filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = sd_models.read_metadata_from_safetensors(filename)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.ssmd_cover_images = self.metadata.pop('ssmd_cover_images', None)  # those are cover images and they are too big to display in UI as text


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None


def assign_lora_names_to_compvis_modules(sd_model):
    lora_layer_mapping = {}

    for name, module in shared.sd_model.cond_stage_model.wrapped.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    for name, module in shared.sd_model.model.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    sd_model.lora_layer_mapping = lora_layer_mapping


def load_lora(name, filename):
    lora = LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = sd_models.read_state_dict(filename)

    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in shared.sd_model.lora_layer_mapping

    for key_diffusers, weight in sd.items():
        key_diffusers_without_lora_parts, lora_key = key_diffusers.split(".", 1)
        key = convert_diffusers_name_to_compvis(key_diffusers_without_lora_parts, is_sd2)

        sd_module = shared.sd_model.lora_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.lora_layer_mapping.get(m.group(1), None)

        if sd_module is None:
            keys_failed_to_match[key_diffusers] = key
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            print(f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}')
            continue
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora


def load_loras(names, multipliers=None):
    already_loaded = {}

    for lora in loaded_loras:
        if lora.name in names:
            already_loaded[lora.name] = lora

    loaded_loras.clear()

    loras_on_disk = [available_loras.get(name, None) for name in names]
    if any([x is None for x in loras_on_disk]):
        list_available_loras()

        loras_on_disk = [available_loras.get(name, None) for name in names]

    for i, name in enumerate(names):
        lora = already_loaded.get(name, None)

        lora_on_disk = loras_on_disk[i]
        if lora_on_disk is not None:
            if lora is None or os.path.getmtime(lora_on_disk.filename) > lora.mtime:
                try:
                    lora = load_lora(name, lora_on_disk.filename)
                except Exception as e:
                    errors.display(e, f"loading Lora {lora_on_disk.filename}")
                    continue

        if lora is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lora.multiplier = multipliers[i] if multipliers else 1.0
        loaded_loras.append(lora)


def lora_calc_updown(lora, module, target):
    with torch.no_grad():
        up = module.up.weight.to(target.device, dtype=target.dtype)
        down = module.down.weight.to(target.device, dtype=target.dtype)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            updown = up @ down

        updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

        return updown


def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Loras to the weights of torch layer self.
    If weights already have this particular set of loras applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to loras.
    """

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in loaded_loras)

    weights_backup = getattr(self, "lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lora in loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += lora_calc_updown(lora, module, self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lora_calc_updown(lora, module_q, self.in_proj_weight)
                updown_k = lora_calc_updown(lora, module_k, self.in_proj_weight)
                updown_v = lora_calc_updown(lora, module_v, self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lora_calc_updown(lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(f'failed to calculate lora weights for layer {lora_layer_name}')

        setattr(self, "lora_current_names", wanted_names)


def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lora_current_names", ())
    setattr(self, "lora_weights_backup", None)


def lora_Linear_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Linear_forward_before_lora(self, input)


def lora_Linear_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lora(self, *args, **kwargs)


def lora_Conv2d_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lora(self, input)


def lora_Conv2d_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_forward(self, *args, **kwargs):
    lora_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lora(self, *args, **kwargs)


def list_available_loras():
    available_loras.clear()

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    candidates = \
        glob.glob(os.path.join(shared.cmd_opts.lora_dir, '**/*.pt'), recursive=True) + \
        glob.glob(os.path.join(shared.cmd_opts.lora_dir, '**/*.safetensors'), recursive=True) + \
        glob.glob(os.path.join(shared.cmd_opts.lora_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_loras[name] = LoraOnDisk(name, filename)


available_loras = {}
loaded_loras = []

list_available_loras()
