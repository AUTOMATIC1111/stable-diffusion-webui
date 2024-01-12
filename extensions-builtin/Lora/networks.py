from typing import Union, List
import os
import re
import time
import concurrent
import lora_patches
import network
import network_lora
import network_hada
import network_ia3
import network_oft
import network_lokr
import network_full
import network_norm
import network_glora
import lora_convert
import torch
import diffusers.models.lora
from modules import shared, devices, sd_models, sd_models_compile, errors, scripts


debug = os.environ.get('SD_LORA_DEBUG', None) is not None
originals: lora_patches.LoraPatches = None
extra_network_lora = None
available_networks = {}
available_network_aliases = {}
loaded_networks: List[network.Network] = []
timer = { 'load': 0, 'apply': 0, 'restore': 0 }
# networks_in_memory = {}
lora_cache = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}
re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")
module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_oft.ModuleTypeOFT(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
]
convert_diffusers_name_to_compvis = lora_convert.convert_diffusers_name_to_compvis # supermerger compatibility item


def assign_network_names_to_compvis_modules(sd_model):
    network_layer_mapping = {}
    if shared.backend == shared.Backend.DIFFUSERS:
        if not hasattr(shared.sd_model, 'text_encoder') or not hasattr(shared.sd_model, 'unet'):
            return
        for name, module in shared.sd_model.text_encoder.named_modules():
            prefix = "lora_te1_" if shared.sd_model_type == "sdxl" else "lora_te_"
            network_name = prefix + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
        if shared.sd_model_type == "sdxl":
            for name, module in shared.sd_model.text_encoder_2.named_modules():
                network_name = "lora_te2_" + name.replace(".", "_")
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name
        for name, module in shared.sd_model.unet.named_modules():
            network_name = "lora_unet_" + name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    else:
        if not hasattr(shared.sd_model, 'cond_stage_model'):
            return
        for name, module in shared.sd_model.cond_stage_model.wrapped.named_modules():
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
        for name, module in shared.sd_model.model.named_modules():
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    sd_model.network_layer_mapping = network_layer_mapping


def load_diffusers(name, network_on_disk, lora_scale=1.0) -> network.Network:
    t0 = time.time()
    cached = lora_cache.get(name, None)
    # if debug:
    shared.log.debug(f'LoRA load: name="{name}" file="{network_on_disk.filename}" type=diffusers {"cached" if cached else ""} fuse={shared.opts.lora_fuse_diffusers}')
    if cached is not None:
        return cached
    if shared.backend != shared.Backend.DIFFUSERS:
        return None
    shared.sd_model.load_lora_weights(network_on_disk.filename)
    if shared.opts.lora_fuse_diffusers:
        shared.sd_model.fuse_lora(lora_scale=lora_scale)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    lora_cache[name] = net
    t1 = time.time()
    timer['load'] += t1 - t0
    return net


def load_network(name, network_on_disk) -> network.Network:
    t0 = time.time()
    cached = lora_cache.get(name, None)
    if debug:
        shared.log.debug(f'LoRA load: name="{name}" file="{network_on_disk.filename}" type=lora {"cached" if cached else ""}')
    if cached is not None:
        return cached
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    sd = sd_models.read_state_dict(network_on_disk.filename)
    assign_network_names_to_compvis_modules(shared.sd_model) # this should not be needed but is here as an emergency fix for an unknown error people are experiencing in 1.2.0
    keys_failed_to_match = {}
    matched_networks = {}
    convert = lora_convert.KeyConvert()
    for key_network, weight in sd.items():
        parts = key_network.split('.')
        if len(parts) > 5: # messy handler for diffusers peft lora
            key_network_without_network_parts = '_'.join(parts[:-2])
            if not key_network_without_network_parts.startswith('lora_'):
                key_network_without_network_parts = 'lora_' + key_network_without_network_parts
            network_part = '.'.join(parts[-2:]).replace('lora_A', 'lora_down').replace('lora_B', 'lora_up')
        else:
            key_network_without_network_parts, network_part = key_network.split(".", 1)
        # if debug:
        #     shared.log.debug(f'LoRA load: name="{name}" full={key_network} network={network_part} key={key_network_without_network_parts}')
        key, sd_module = convert(key_network_without_network_parts)
        if sd_module is None:
            keys_failed_to_match[key_network] = key
            continue
        if key not in matched_networks:
            matched_networks[key] = network.NetworkWeights(network_key=key_network, sd_key=key, w={}, sd_module=sd_module)
        matched_networks[key].w[network_part] = weight
    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                break
        if net_module is None:
            shared.log.error(f'LoRA unhandled: name={name} key={key} weights={weights.w.keys()}')
        else:
            net.modules[key] = net_module
    if len(keys_failed_to_match) > 0:
        shared.log.warning(f"LoRA file={network_on_disk.filename} unmatched={len(keys_failed_to_match)} matched={len(matched_networks)}")
        if debug:
            shared.log.debug(f"LoRA file={network_on_disk.filename} unmatched={keys_failed_to_match}")
    elif debug:
        shared.log.debug(f"LoRA file={network_on_disk.filename} unmatched={len(keys_failed_to_match)} matched={len(matched_networks)}")
    lora_cache[name] = net
    t1 = time.time()
    timer['load'] += t1 - t0
    return net


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    networks_on_disk = [available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_network_aliases.get(name, None) for name in names]
    failed_to_load_networks = []

    recompile_model = False
    if shared.compiled_model_state is not None and shared.compiled_model_state.is_compiled:
        if len(names) == len(shared.compiled_model_state.lora_model):
            for i, name in enumerate(names):
                if shared.compiled_model_state.lora_model[i] != f"{name}:{te_multipliers[i] if te_multipliers else 1.0}":
                    recompile_model = True
                    shared.compiled_model_state.lora_model = []
                    break
            if not recompile_model:
                if len(loaded_networks) > 0 and debug:
                    shared.log.debug('Model Compile: Skipping LoRa loading')
                return
        else:
            recompile_model = True
            shared.compiled_model_state.lora_model = []
    if recompile_model:
        backup_cuda_compile = shared.opts.cuda_compile
        backup_nncf_compress_weights = shared.opts.nncf_compress_weights
        backup_nncf_compress_text_encoder_weights = shared.opts.nncf_compress_text_encoder_weights
        sd_models.unload_model_weights(op='model')
        shared.opts.cuda_compile = False
        shared.opts.nncf_compress_weights = False
        shared.opts.nncf_compress_text_encoder_weights = False
        sd_models.reload_model_weights(op='model')
        shared.opts.cuda_compile = backup_cuda_compile
        shared.opts.nncf_compress_weights = backup_nncf_compress_weights
        shared.opts.nncf_compress_text_encoder_weights = backup_nncf_compress_text_encoder_weights

    loaded_networks.clear()
    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = None
        if network_on_disk is not None:
            if debug:
                shared.log.debug(f'LoRA load start: name="{name}" file="{network_on_disk.filename}"')
            try:
                if recompile_model:
                    shared.compiled_model_state.lora_model.append(f"{name}:{te_multipliers[i] if te_multipliers else 1.0}")
                if shared.backend == shared.Backend.DIFFUSERS and shared.opts.lora_force_diffusers: # OpenVINO only works with Diffusers LoRa loading.
                    # or getattr(network_on_disk, 'shorthash', '').lower() == 'aaebf6360f7d' # sd15-lcm
                    # or getattr(network_on_disk, 'shorthash', '').lower() == '3d18b05e4f56' # sdxl-lcm
                    # or getattr(network_on_disk, 'shorthash', '').lower() == '813ea5fb1c67' # turbo sdxl-turbo
                    net = load_diffusers(name, network_on_disk, lora_scale=te_multipliers[i] if te_multipliers else 1.0)
                else:
                    net = load_network(name, network_on_disk)
            except Exception as e:
                shared.log.error(f"LoRA load failed: file={network_on_disk.filename} {e}")
                if debug:
                    errors.display(e, f"LoRA load failed file={network_on_disk.filename}")
                continue
            net.mentioned_name = name
            network_on_disk.read_hash()
        if net is None:
            failed_to_load_networks.append(name)
            shared.log.error(f"LoRA unknown type: network={name}")
            continue
        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        loaded_networks.append(net)

    while len(lora_cache) > shared.opts.lora_in_memory_limit:
        name = next(iter(lora_cache))
        lora_cache.pop(name, None)
    if len(loaded_networks) > 0 and debug:
        shared.log.debug(f'LoRA loaded={len(loaded_networks)} cache={list(lora_cache)}')
    devices.torch_gc()

    if recompile_model:
        shared.log.info("LoRA recompiling model")
        backup_lora_model = shared.compiled_model_state.lora_model
        sd_models_compile.compile_diffusers(shared.sd_model)
        shared.compiled_model_state.lora_model = backup_lora_model


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv]):
    t0 = time.time()
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)
    if weights_backup is None and bias_backup is None:
        return
    # if debug:
    #     shared.log.debug('LoRA restore weights')
    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.in_proj_weight.copy_(weights_backup[0])
            self.out_proj.weight.copy_(weights_backup[1])
        else:
            self.weight.copy_(weights_backup)
    if bias_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias.copy_(bias_backup)
        else:
            self.bias.copy_(bias_backup)
    else:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias = None
        else:
            self.bias = None
    t1 = time.time()
    timer['restore'] += t1 - t0


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to networks.
    """
    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return
    t0 = time.time()
    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)
    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != (): # pylint: disable=C1803
        if current_names != ():
            raise RuntimeError("no backup weights found and current weights are not unchanged")
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)
        self.network_weights_backup = weights_backup
    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_backup = self.out_proj.bias.to(devices.cpu, copy=True)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = self.bias.to(devices.cpu, copy=True)
        else:
            bias_backup = None
        self.network_bias_backup = bias_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)
        for net in loaded_networks:
            # default workflow where module is known and has weights
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                try:
                    with devices.inference_context():
                        updown, ex_bias = module.calc_updown(self.weight)
                        if len(self.weight.shape) == 4 and self.weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5)) # pylint: disable=not-callable
                        self.weight += updown
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias)
                            else:
                                self.bias += ex_bias
                except RuntimeError as e:
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                    if debug:
                        module_name = net.modules.get(network_layer_name, None)
                        shared.log.error(f"LoRA apply weight name={net.name} module={module_name} layer={network_layer_name} {e}")
                        errors.display(e, 'LoRA apply weight')
                        raise RuntimeError('LoRA apply weight') from e
                continue
            # alternative workflow looking at _*_proj layers
            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)
            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with devices.inference_context():
                        updown_q, _ = module_q.calc_updown(self.in_proj_weight)
                        updown_k, _ = module_k.calc_updown(self.in_proj_weight)
                        updown_v, _ = module_v.calc_updown(self.in_proj_weight)
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight)
                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                    if ex_bias is not None:
                        if self.out_proj.bias is None:
                            self.out_proj.bias = torch.nn.Parameter(ex_bias)
                        else:
                            self.out_proj.bias += ex_bias
                except RuntimeError as e:
                    if debug:
                        shared.log.debug(f"LoRA network={net.name} layer={network_layer_name} {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                continue
            if module is None:
                continue
            shared.log.warning(f"LoRA network={net.name} layer={network_layer_name} unsupported operation")
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
        self.network_current_names = wanted_names
    t1 = time.time()
    timer['apply'] += t1 - t0


def network_forward(module, input, original_forward): # pylint: disable=W0622
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """
    if len(loaded_networks) == 0:
        return original_forward(module, input)
    input = devices.cond_cast_unet(input)
    network_restore_weights_from_backup(module)
    network_reset_cached_weight(module)
    y = original_forward(module, input)
    network_layer_name = getattr(module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue
        y = module.forward(input, y)
    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None


def network_Linear_forward(self, input): # pylint: disable=W0622
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Linear_forward)
    network_apply_weights(self)
    return originals.Linear_forward(self, input)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.Linear_load_state_dict(self, *args, **kwargs)


def network_Conv2d_forward(self, input): # pylint: disable=W0622
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Conv2d_forward)
    network_apply_weights(self)
    return originals.Conv2d_forward(self, input)


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


def network_GroupNorm_forward(self, input): # pylint: disable=W0622
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.GroupNorm_forward)
    network_apply_weights(self)
    return originals.GroupNorm_forward(self, input)


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


def network_LayerNorm_forward(self, input): # pylint: disable=W0622
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.LayerNorm_forward)
    network_apply_weights(self)
    return originals.LayerNorm_forward(self, input)


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)
    return originals.MultiheadAttention_forward(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})
    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)
    candidates = []
    if os.path.exists(shared.cmd_opts.lora_dir):
        candidates += list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    else:
        shared.log.warning('LoRA directory not found: path="{shared.cmd_opts.lora_dir}"')
    if os.path.exists(shared.cmd_opts.lyco_dir):
        candidates += list(shared.walk_files(shared.cmd_opts.lyco_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))

    def add_network(filename):
        if os.path.isdir(filename):
            return
        name = os.path.splitext(os.path.basename(filename))[0]
        try:
            entry = network.NetworkOnDisk(name, filename)
            available_networks[entry.name] = entry
            if entry.alias in available_network_aliases:
                forbidden_network_aliases[entry.alias.lower()] = 1
            available_network_aliases[entry.name] = entry
            available_network_aliases[entry.alias] = entry
            if entry.shorthash:
                available_network_hash_lookup[entry.shorthash] = entry
        except OSError as e:  # should catch FileNotFoundError and PermissionError etc.
            shared.log.error(f"Failed to load network {name} from {filename} {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
        for fn in candidates:
            executor.submit(add_network, fn)


def infotext_pasted(infotext, params): # pylint: disable=W0613
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything
    added = []
    for k in params:
        if not k.startswith("AddNet Model "):
            continue
        num = k[13:]
        if params.get("AddNet Module " + num) != "LoRA":
            continue
        name = params.get("AddNet Model " + num)
        if name is None:
            continue
        m = re_network_name.match(name)
        if m:
            name = m.group(1)
        multiplier = params.get("AddNet Weight A " + num, "1.0")
        added.append(f"<lora:{name}:{multiplier}>")
    if added:
        params["Prompt"] += "\n" + "".join(added)


list_available_networks()
