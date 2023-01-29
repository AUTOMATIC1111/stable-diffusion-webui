import glob
import os
import re
import torch

from modules import shared, devices, sd_models

re_digits = re.compile(r"\d+")
re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")
re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")


def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename


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

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        fullkey = convert_diffusers_name_to_compvis(key_diffusers)
        key, lora_key = fullkey.split(".", 1)

        sd_module = shared.sd_model.lora_layer_mapping.get(key, None)
        if sd_module is None:
            keys_failed_to_match.append(key_diffusers)
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
        elif type(sd_module) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.device, dtype=devices.dtype)

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
                lora = load_lora(name, lora_on_disk.filename)

        if lora is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lora.multiplier = multipliers[i] if multipliers else 1.0
        loaded_loras.append(lora)


def lora_forward(module, input, res):
    if len(loaded_loras) == 0:
        return res

    lora_layer_name = getattr(module, 'lora_layer_name', None)
    for lora in loaded_loras:
        module = lora.modules.get(lora_layer_name, None)
        if module is not None:
            if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
                res = res + module.up(module.down(res)) * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)
            else:
                res = res + module.up(module.down(input)) * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

    return res


def lora_Linear_forward(self, input):
    return lora_forward(self, input, torch.nn.Linear_forward_before_lora(self, input))


def lora_Conv2d_forward(self, input):
    return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora(self, input))


def list_available_loras():
    available_loras.clear()

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    candidates = \
        glob.glob(os.path.join(shared.cmd_opts.lora_dir, '**/*.pt'), recursive=True) + \
        glob.glob(os.path.join(shared.cmd_opts.lora_dir, '**/*.safetensors'), recursive=True) + \
        glob.glob(os.path.join(shared.cmd_opts.lora_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_loras[name] = LoraOnDisk(name, filename)


available_loras = {}
loaded_loras = []

list_available_loras()
