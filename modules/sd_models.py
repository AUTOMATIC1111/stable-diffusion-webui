import re
import io
import sys
import json
import time
import copy
import logging
import threading
import contextlib
import collections
import os.path
from os import mkdir
from urllib import request
from enum import Enum
from rich import progress # pylint: disable=redefined-builtin
import torch
import safetensors.torch
from omegaconf import OmegaConf
import tomesd
from transformers import logging as transformers_logging
import ldm.modules.midas as midas
from ldm.util import instantiate_from_config
from modules import paths, shared, shared_items, shared_state, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config
from modules.sd_hijack_inpainting import do_inpainting_hijack
from modules.timer import Timer
from modules.memstats import memory_stats
from modules.paths_internal import models_path, script_path

try:
    import diffusers
except Exception as ex:
    shared.log.error(f'Failed to import diffusers: {ex}')


transformers_logging.set_verbosity_error()
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
checkpoints_list = {}
checkpoint_aliases = {}
checkpoints_loaded = collections.OrderedDict()
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0
sd_metadata_timer = 0


class CheckpointInfo:
    def __init__(self, filename):
        self.name = None
        self.hash = None
        self.filename = filename
        self.type = ''
        relname = filename
        app_path = os.path.abspath(script_path)

        def rel(fn, path):
            try:
                return os.path.relpath(fn, path)
            except Exception:
                return fn

        if relname.startswith('..'):
            relname = os.path.abspath(relname)
        if relname.startswith(shared.opts.ckpt_dir):
            relname = rel(filename, shared.opts.ckpt_dir)
        elif relname.startswith(shared.opts.diffusers_dir):
            relname = rel(filename, shared.opts.diffusers_dir)
        elif relname.startswith(model_path):
            relname = rel(filename, model_path)
        elif relname.startswith(script_path):
            relname = rel(filename, script_path)
        elif relname.startswith(app_path):
            relname = rel(filename, app_path)
        else:
            relname = os.path.abspath(relname)
        relname, ext = os.path.splitext(relname)
        ext = ext.lower()[1:]

        if os.path.isfile(filename): # ckpt or safetensor
            self.name = relname
            self.filename = filename
            self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{relname}")
            self.type = ext
            # self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        else: # maybe a diffuser
            repo = [r for r in modelloader.diffuser_repos if filename == r['filename']]
            if len(repo) == 0:
                self.name = relname
                self.filename = filename
                self.sha256 = None
                self.type = 'unknown'
            else:
                self.name = os.path.join(os.path.basename(shared.opts.diffusers_dir), repo[0]['name'])
                self.filename = repo[0]['path']
                self.sha256 = repo[0]['hash']
                self.type = 'diffusers'

        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.title = self.name if self.shorthash is None else f'{self.name} [{self.shorthash}]'
        self.path = self.filename
        self.model_name = os.path.basename(self.name)
        # shared.log.debug(f'Checkpoint: type={self.type} name={self.name} filename={self.filename} hash={self.shorthash} title={self.title}')
        self.metadata = read_metadata_from_safetensors(filename)

    def register(self):
        checkpoints_list[self.title] = self
        for i in [self.name, self.filename, self.shorthash, self.title]:
            if i is not None:
                checkpoint_aliases[i] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return None
        self.shorthash = self.sha256[0:10]
        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()
        return self.shorthash


#Used by OpenVINO, can be used with TensorRT or Olive
class CompiledModelState:
    def __init__(self):
        self.first_pass = True
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.partition_id = 0
        self.cn_model = []
        self.lora_model = []


class NoWatermark:
    def apply_watermark(self, img):
        return img


def setup_model():
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    list_models()
    enable_midas_autodownload()


def checkpoint_tiles(use_short=False): # pylint: disable=unused-argument
    def convert(name):
        return int(name) if name.isdigit() else name.lower()
    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    t0 = time.time()
    global checkpoints_list # pylint: disable=global-statement
    checkpoints_list.clear()
    checkpoint_aliases.clear()
    if shared.opts.sd_disable_ckpt or shared.backend == shared.Backend.DIFFUSERS:
        ext_filter = [".safetensors"]
    else:
        ext_filter = [".ckpt", ".safetensors"]
    model_list = modelloader.load_models(model_path=model_path, model_url=None, command_path=shared.opts.ckpt_dir, ext_filter=ext_filter, download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
    if shared.backend == shared.Backend.DIFFUSERS:
        model_list += modelloader.load_diffusers_models(model_path=os.path.join(models_path, 'Diffusers'), command_path=shared.opts.diffusers_dir)
    for filename in sorted(model_list, key=str.lower):
        checkpoint_info = CheckpointInfo(filename)
        if checkpoint_info.name is not None:
            checkpoint_info.register()
    if shared.cmd_opts.ckpt is not None:
        if not os.path.exists(shared.cmd_opts.ckpt) and shared.backend == shared.Backend.ORIGINAL:
            if shared.cmd_opts.ckpt.lower() != "none":
                shared.log.warning(f"Requested checkpoint not found: {shared.cmd_opts.ckpt}")
        else:
            checkpoint_info = CheckpointInfo(shared.cmd_opts.ckpt)
            if checkpoint_info.name is not None:
                checkpoint_info.register()
                shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif shared.cmd_opts.ckpt != shared.default_sd_model_file and shared.cmd_opts.ckpt is not None:
        shared.log.warning(f"Checkpoint not found: {shared.cmd_opts.ckpt}")
    shared.log.info(f'Available models: path="{shared.opts.ckpt_dir}" items={len(checkpoints_list)} time={time.time()-t0:.2f}s')

    checkpoints_list = dict(sorted(checkpoints_list.items(), key=lambda cp: cp[1].filename))
    if len(checkpoints_list) == 0:
        if not shared.cmd_opts.no_download:
            key = input('Download the default model? (y/N) ')
            if key.lower().startswith('y'):
                if shared.backend == shared.Backend.ORIGINAL:
                    model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
                    shared.opts.data['sd_model_checkpoint'] = "v1-5-pruned-emaonly.safetensors"
                    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
                else:
                    default_model_id = "runwayml/stable-diffusion-v1-5"
                    modelloader.download_diffusers_model(default_model_id, shared.opts.diffusers_dir)
                    model_list = modelloader.load_diffusers_models(model_path=os.path.join(models_path, 'Diffusers'), command_path=shared.opts.diffusers_dir)

                for filename in sorted(model_list, key=str.lower):
                    checkpoint_info = CheckpointInfo(filename)
                    if checkpoint_info.name is not None:
                        checkpoint_info.register()


def update_model_hashes():
    txt = []
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.hash is None]
    shared.log.info(f'Models list: short hash missing for {len(lst)} out of {len(checkpoints_list)} models')
    for ckpt in lst:
        ckpt.hash = model_hash(ckpt.filename)
        txt.append(f'Calculated short hash: <b>{ckpt.title}</b> {ckpt.hash}')
    txt.append(f'Updated short hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.sha256 is None or ckpt.shorthash is None]
    shared.log.info(f'Models list: full hash missing for {len(lst)} out of {len(checkpoints_list)} models')
    for ckpt in lst:
        ckpt.sha256 = hashes.sha256(ckpt.filename, f"checkpoint/{ckpt.name}")
        ckpt.shorthash = ckpt.sha256[0:10] if ckpt.sha256 is not None else None
        if ckpt.sha256 is not None:
            txt.append(f'Calculated full hash: <b>{ckpt.title}</b> {ckpt.shorthash}')
        else:
            txt.append(f'Skipped hash calculation: <b>{ckpt.title}</b>')
    txt.append(f'Updated hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
    txt = '<br>'.join(txt)
    return txt


def get_closet_checkpoint_match(search_string):
    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info
    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]
    found = sorted([info for info in checkpoints_list.values() if search_string.split(' ')[0] in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]
    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""
    try:
        with open(filename, "rb") as file:
            import hashlib
            # t0 = time.time()
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            shorthash = m.hexdigest()[0:8]
            # t1 = time.time()
            # shared.log.debug(f'Calculating short hash: {filename} hash={shorthash} time={(t1-t0):.2f}')
            return shorthash
    except FileNotFoundError:
        return 'NOFILE'
    except Exception:
        return 'NOHASH'


def select_checkpoint(op='model'):
    if op == 'dict':
        model_checkpoint = shared.opts.sd_model_dict
    elif op == 'refiner':
        model_checkpoint = shared.opts.data.get('sd_model_refiner', None)
    else:
        model_checkpoint = shared.opts.sd_model_checkpoint
    if model_checkpoint is None or model_checkpoint == 'None':
        return None
    checkpoint_info = get_closet_checkpoint_match(model_checkpoint)
    if checkpoint_info is not None:
        shared.log.info(f'Select: {op}="{checkpoint_info.title if checkpoint_info is not None else None}"')
        return checkpoint_info
    if len(checkpoints_list) == 0 and not shared.cmd_opts.no_download:
        shared.log.error("Cannot generate without a checkpoint")
        shared.log.info("Set system paths to use existing folders in a different location")
        shared.log.info("Or use --ckpt <path-to-checkpoint> to force using existing checkpoint")
        return None
    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        if model_checkpoint != 'model.ckpt' and model_checkpoint != 'runwayml/stable-diffusion-v1-5':
            shared.log.warning(f"Selected checkpoint not found: {model_checkpoint}")
        else:
            shared.log.info("Selecting first available checkpoint")
        # shared.log.warning(f"Loading fallback checkpoint: {checkpoint_info.title}")
        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    shared.log.info(f'Select: {op}="{checkpoint_info.title if checkpoint_info is not None else None}"')
    return checkpoint_info


checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


def write_metadata():
    global sd_metadata_pending # pylint: disable=global-statement
    if sd_metadata_pending == 0:
        shared.log.debug(f"Model metadata: {sd_metadata_file} no changes")
        return
    shared.writefile(sd_metadata, sd_metadata_file)
    shared.log.info(f"Model metadata saved: {sd_metadata_file} items={sd_metadata_pending} time={sd_metadata_timer:.2f}s")
    sd_metadata_pending = 0


def scrub_dict(dict_obj, keys):
    for key in list(dict_obj.keys()):
        if not isinstance(dict_obj, dict):
            continue
        if key in keys:
            dict_obj.pop(key, None)
        elif isinstance(dict_obj[key], dict):
            scrub_dict(dict_obj[key], keys)
        elif isinstance(dict_obj[key], list):
            for item in dict_obj[key]:
                scrub_dict(item, keys)


def read_metadata_from_safetensors(filename):
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        if not os.path.isfile(sd_metadata_file):
            sd_metadata = {}
        else:
            sd_metadata = shared.readfile(sd_metadata_file)
    res = sd_metadata.get(filename, None)
    if res is not None:
        return res
    if not filename.endswith(".safetensors"):
        return {}
    if shared.cmd_opts.no_metadata:
        return {}
    res = {}
    try:
        t0 = time.time()
        with open(filename, mode="rb") as file:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)
            if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
                shared.log.error(f"Not a valid safetensors file: {filename}")
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                if v.startswith("data:"):
                    v = 'data'
                if k == 'format' and v == 'pt':
                    continue
                large = True if len(v) > 2048 else False
                if large and k == 'ss_datasets':
                    continue
                if large and k == 'workflow':
                    continue
                if large and k == 'prompt':
                    continue
                if large and k == 'ss_bucket_info':
                    continue
                if v[0:1] == '{':
                    try:
                        v = json.loads(v)
                        if large and k == 'ss_tag_frequency':
                            v = { i: len(j) for i, j in v.items() }
                        if large and k == 'sd_merge_models':
                            scrub_dict(v, ['sd_merge_recipe'])
                    except Exception:
                        pass
                res[k] = v
        sd_metadata[filename] = res
        global sd_metadata_pending # pylint: disable=global-statement
        sd_metadata_pending += 1
        t1 = time.time()
        global sd_metadata_timer # pylint: disable=global-statement
        sd_metadata_timer += (t1 - t0)
    except Exception as e:
        shared.log.error(f"Error reading metadata from: {filename} {e}")
    return res


def read_state_dict(checkpoint_file, map_location=None): # pylint: disable=unused-argument
    if not os.path.isfile(checkpoint_file):
        shared.log.error(f"Model is not a file: {checkpoint_file}")
        return None
    try:
        pl_sd = None
        with progress.open(checkpoint_file, 'rb', description=f'[cyan]Loading weights: [yellow]{checkpoint_file}', auto_refresh=True, console=shared.console) as f:
            _, extension = os.path.splitext(checkpoint_file)
            if extension.lower() == ".ckpt" and shared.opts.sd_disable_ckpt:
                shared.log.warning(f"Checkpoint loading disabled: {checkpoint_file}")
                return None
            if shared.opts.stream_load:
                if extension.lower() == ".safetensors":
                    # shared.log.debug('Model weights loading: type=safetensors mode=buffered')
                    buffer = f.read()
                    pl_sd = safetensors.torch.load(buffer)
                else:
                    # shared.log.debug('Model weights loading: type=checkpoint mode=buffered')
                    buffer = io.BytesIO(f.read())
                    pl_sd = torch.load(buffer, map_location='cpu')
            else:
                if extension.lower() == ".safetensors":
                    # shared.log.debug('Model weights loading: type=safetensors mode=mmap')
                    pl_sd = safetensors.torch.load_file(checkpoint_file, device='cpu')
                else:
                    # shared.log.debug('Model weights loading: type=checkpoint mode=direct')
                    pl_sd = torch.load(f, map_location='cpu')
            sd = get_state_dict_from_checkpoint(pl_sd)
        del pl_sd
    except Exception as e:
        errors.display(e, f'loading model: {checkpoint_file}')
        sd = None
    return sd


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    if not os.path.isfile(checkpoint_info.filename):
        return None
    if checkpoint_info in checkpoints_loaded:
        shared.log.info("Model weights loading: from cache")
        return checkpoints_loaded[checkpoint_info]
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load")
    return res


def load_model_weights(model: torch.nn.Module, checkpoint_info: CheckpointInfo, state_dict, timer):
    _pipeline, _model_type = detect_pipeline(checkpoint_info.path, 'model')
    shared.log.debug(f'Model weights loading: {memory_stats()}')
    timer.record("hash")
    if model_data.sd_dict == 'None':
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        shared.log.error(f'Error loading model weights: {checkpoint_info.filename}')
        shared.log.error(' '.join(str(e).splitlines()[:2]))
        return False
    del state_dict
    timer.record("apply")
    if shared.opts.sd_checkpoint_cache > 0:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = model.state_dict().copy()
    if shared.opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("channels")
    if not shared.opts.no_half:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)
        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.opts.no_half_vae:
            model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to float16
        if shared.opts.upcast_sampling and depth_model:
            model.depth_model = None
        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model
    if shared.opts.cuda_cast_unet:
        devices.dtype_unet = model.model.diffusion_model.dtype
    else:
        model.model.diffusion_model.to(devices.dtype_unet)
    model.first_stage_model.to(devices.dtype_vae)
    # clean up cache if limit is reached
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)
    model.sd_model_hash = checkpoint_info.calculate_shorthash()
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
    model.logvar = model.logvar.to(devices.device)  # fix for training
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("vae")
    return True


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """
    midas_path = os.path.join(paths.models_path, 'midas')
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)
    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }
    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                mkdir(midas_path)
            shared.log.info(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            shared.log.info(f"{model_type} downloaded")
        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def repair_config(sd_config):
    if "use_ema" not in sd_config.model.params:
        sd_config.model.params.use_ema = False
    if shared.opts.no_half:
        sd_config.model.params.unet_config.params.use_fp16 = False
    elif shared.opts.upcast_sampling:
        sd_config.model.params.unet_config.params.use_fp16 = True if sys.platform != 'darwin' else False
    if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
        sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"
    # For UnCLIP-L, override the hardcoded karlo directory
    if "noise_aug_config" in sd_config.model.params and "clip_stats_path" in sd_config.model.params.noise_aug_config.params:
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'


class ModelData:
    def __init__(self):
        self.sd_model = None
        self.sd_refiner = None
        self.sd_dict = 'None'
        self.initial = True
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.sd_model is None and shared.opts.sd_model_checkpoint != 'None' and not self.lock.locked():
            with self.lock:
                try:
                    self.sd_model = reload_model_weights(op='model')
                    if self.sd_model is not None:
                        self.sd_model.is_sdxl = False # a1111 compatibility item
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_model = None
        return self.sd_model

    def set_sd_model(self, v):
        self.sd_model = v

    def get_sd_refiner(self):
        if self.sd_refiner is None and shared.opts.sd_model_refiner != 'None' and not self.lock.locked():
            with self.lock:
                try:
                    self.sd_refiner = reload_model_weights(op='refiner')
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_refiner = None
        return self.sd_refiner

    def set_sd_refiner(self, v):
        self.sd_refiner = v

model_data = ModelData()


def change_backend():
    shared.log.info(f'Backend changed: {shared.backend}')
    shared.log.warning('Server restart required to apply all changes')
    if shared.backend == shared.Backend.ORIGINAL:
        change_from = shared.Backend.DIFFUSERS
    else:
        change_from = shared.Backend.ORIGINAL
    unload_model_weights(change_from=change_from)
    checkpoints_loaded.clear()
    from modules.sd_samplers import list_samplers
    list_samplers(shared.backend)
    list_models()
    from modules.sd_vae import refresh_vae_list
    refresh_vae_list()


def detect_pipeline(f: str, op: str = 'model'):
    if not f.endswith('.safetensors'):
        return None, None
    guess = shared.opts.diffusers_pipeline
    if guess == 'Autodetect':
        try:
            size = round(os.path.getsize(f) / 1024 / 1024)
            if size < 128:
                shared.log.warning(f'Model size smaller than expected: {f} size={size} MB')
            elif (size >= 316 and size <= 324) or (size >= 156 and size <= 164): # 320 or 160
                shared.log.warning(f'Model detected as VAE model, but attempting to load as model: {op}={f} size={size} MB')
                guess = 'VAE'
            elif size >= 5351 and size <= 5359: # 5353
                guess = 'Stable Diffusion' # SD v2
            elif size >= 5791 and size <= 5799: # 5795
                if shared.backend == shared.Backend.ORIGINAL:
                    shared.log.warning(f'Model detected as SD-XL refiner model, but attempting to load using backend=original: {op}={f} size={size} MB')
                if op == 'model':
                    shared.log.warning(f'Model detected as SD-XL refiner model, but attempting to load a base model: {op}={f} size={size} MB')
                guess = 'Stable Diffusion XL'
            elif (size >= 6611 and size <= 6619) or (size >= 6771 and size <= 6779): # 6617, HassakuXL is 6776
                if shared.backend == shared.Backend.ORIGINAL:
                    shared.log.warning(f'Model detected as SD-XL base model, but attempting to load using backend=original: {op}={f} size={size} MB')
                guess = 'Stable Diffusion XL'
            elif size >= 3361 and size <= 3369: # 3368
                if shared.backend == shared.Backend.ORIGINAL:
                    shared.log.warning(f'Model detected as SD upscale model, but attempting to load using backend=original: {op}={f} size={size} MB')
                guess = 'Stable Diffusion Upscale'
            elif size >= 4891 and size <= 4899: # 4897
                if shared.backend == shared.Backend.ORIGINAL:
                    shared.log.warning(f'Model detected as SD XL inpaint model, but attempting to load using backend=original: {op}={f} size={size} MB')
                guess = 'Stable Diffusion XL Inpaint'
            elif size >= 9791 and size <= 9799: # 9794
                if shared.backend == shared.Backend.ORIGINAL:
                    shared.log.warning(f'Model detected as SD XL instruct pix2pix model, but attempting to load using backend=original: {op}={f} size={size} MB')
                guess = 'Stable Diffusion XL Instruct'
            else:
                guess = 'Stable Diffusion'
            pipeline = shared_items.get_pipelines().get(guess, None)
            shared.log.info(f'Autodetect: {op}="{guess}" class={pipeline.__name__} file="{f}" size={size}MB')
        except Exception as e:
            shared.log.error(f'Error detecting diffusers pipeline: model={f} {e}')
            return None, None
    else:
        try:
            size = round(os.path.getsize(f) / 1024 / 1024)
            pipeline = shared_items.get_pipelines().get(guess, None)
            shared.log.info(f'Diffusers: {op}="{guess}" class={pipeline.__name__} file="{f}" size={size}MB')
        except Exception as e:
            shared.log.error(f'Error loading diffusers pipeline: model={f} {e}')

    if pipeline is None:
        shared.log.warning(f'Autodetect: pipeline not recognized: {guess}: {op}={f} size={size}')
        pipeline = diffusers.StableDiffusionPipeline
    return pipeline, guess


def compile_diffusers(sd_model):
    try:
        if shared.opts.ipex_optimize:
            import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
            sd_model.unet.training = False
            sd_model.unet = ipex.optimize(sd_model.unet, dtype=devices.dtype_unet, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
            if hasattr(sd_model, 'vae'):
                sd_model.vae.training = False
                sd_model.vae = ipex.optimize(sd_model.vae, dtype=devices.dtype_vae, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
            if hasattr(sd_model, 'movq'):
                sd_model.movq.training = False
                sd_model.movq = ipex.optimize(sd_model.movq, dtype=devices.dtype_vae, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
            shared.log.info("Applied IPEX Optimize.")
    except Exception as err:
        shared.log.warning(f"IPEX Optimize not supported: {err}")

    try:
        if shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
            shared.log.info(f"Compiling pipeline={sd_model.__class__.__name__} shape={8 * sd_model.unet.config.sample_size} mode={shared.opts.cuda_compile_backend}")
            import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
            if shared.opts.cuda_compile_backend == "openvino_fx":
                torch._dynamo.reset() # pylint: disable=protected-access
                from modules.intel.openvino import openvino_fx, openvino_clear_caches # pylint: disable=unused-import
                openvino_clear_caches()
                torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True # pylint: disable=protected-access
                if shared.compiled_model_state is None:
                    shared.compiled_model_state = CompiledModelState()
                shared.compiled_model_state.first_pass = True if not shared.opts.cuda_compile_precompile else False
            log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
            if hasattr(torch, '_logging'):
                torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
            torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
            torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access
            sd_model.unet = torch.compile(sd_model.unet, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph) # pylint: disable=attribute-defined-outside-init
            if hasattr(sd_model, 'vae'):
                sd_model.vae.decode = torch.compile(sd_model.vae.decode, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph) # pylint: disable=attribute-defined-outside-init
            if hasattr(sd_model, 'movq'):
                sd_model.movq.decode = torch.compile(sd_model.movq.decode, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph) # pylint: disable=attribute-defined-outside-init
            if shared.opts.cuda_compile_precompile:
                sd_model("dummy prompt")
            shared.log.info("Complilation done.")
    except Exception as err:
        shared.log.warning(f"Model compile not supported: {err}")


def set_diffuser_options(sd_model, vae, op: str):
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return
    if (shared.opts.diffusers_model_cpu_offload or shared.cmd_opts.medvram) and (shared.opts.diffusers_seq_cpu_offload or shared.cmd_opts.lowvram):
        shared.log.warning(f'Setting {op}: Model CPU offload and Sequential CPU offload are not compatible')
        shared.log.debug(f'Setting {op}: disabling model CPU offload')
        shared.opts.diffusers_model_cpu_offload=False
        shared.cmd_opts.medvram=False

    if hasattr(sd_model, "watermark"):
        sd_model.watermark = NoWatermark()
    sd_model.has_accelerate = False
    if hasattr(sd_model, "enable_model_cpu_offload"):
        if (shared.cmd_opts.medvram and devices.backend != "directml") or shared.opts.diffusers_model_cpu_offload:
            shared.log.debug(f'Setting {op}: enable model CPU offload')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Model CPU offload" is enabled')
            sd_model.enable_model_cpu_offload()
            sd_model.has_accelerate = True
    if hasattr(sd_model, "enable_sequential_cpu_offload"):
        if shared.cmd_opts.lowvram or shared.opts.diffusers_seq_cpu_offload:
            shared.log.debug(f'Setting {op}: enable sequential CPU offload')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Sequential CPU offload" is enabled')
            sd_model.enable_sequential_cpu_offload(device=devices.device)
            sd_model.has_accelerate = True
    if hasattr(sd_model, "enable_vae_slicing"):
        if shared.cmd_opts.lowvram or shared.opts.diffusers_vae_slicing:
            shared.log.debug(f'Setting {op}: enable VAE slicing')
            sd_model.enable_vae_slicing()
        else:
            sd_model.disable_vae_slicing()
    if hasattr(sd_model, "enable_vae_tiling"):
        if shared.cmd_opts.lowvram or shared.opts.diffusers_vae_tiling:
            shared.log.debug(f'Setting {op}: enable VAE tiling')
            sd_model.enable_vae_tiling()
        else:
            sd_model.disable_vae_tiling()
    if hasattr(sd_model, "enable_attention_slicing"):
        if shared.cmd_opts.lowvram or shared.opts.diffusers_attention_slicing:
            shared.log.debug(f'Setting {op}: enable attention slicing')
            sd_model.enable_attention_slicing()
        else:
            sd_model.disable_attention_slicing()
    if hasattr(sd_model, "vae"):
        if vae is not None:
            sd_model.vae = vae
        if shared.opts.diffusers_vae_upcast != 'default':
            if shared.opts.diffusers_vae_upcast == 'true':
                # sd_model.vae.config["force_upcast"] = True
                sd_model.vae.config.force_upcast = True
            else:
                # sd_model.vae.config["force_upcast"] = False
                sd_model.vae.config.force_upcast = False
            if shared.opts.no_half_vae:
                devices.dtype_vae = torch.float32
                sd_model.vae.to(devices.dtype_vae)
        shared.log.debug(f'Setting {op} VAE: name={sd_vae.loaded_vae_file} upcast={sd_model.vae.config.get("force_upcast", None)}')
    if shared.opts.cross_attention_optimization == "xFormers" and hasattr(sd_model, 'enable_xformers_memory_efficient_attention'):
        sd_model.enable_xformers_memory_efficient_attention()
    if shared.opts.opt_channelslast:
        shared.log.debug(f'Setting {op}: enable channels last')
        sd_model.unet.to(memory_format=torch.channels_last)


def load_diffuser(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'): # pylint: disable=unused-argument
    import torch # pylint: disable=reimported,redefined-outer-name
    if timer is None:
        timer = Timer()
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    timer.record("diffusers")
    devices.set_cuda_params()
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
        "load_safety_checker": False,
        "load_connected_pipeline": True,
        # TODO: use_safetensors cant enable for all checkpoints just yet
    }
    if shared.opts.diffusers_model_load_variant == 'default':
        if devices.dtype == torch.float16:
            diffusers_load_config['variant'] = 'fp16'
    elif shared.opts.diffusers_model_load_variant == 'fp32':
        pass
    else:
        diffusers_load_config['variant'] = shared.opts.diffusers_model_load_variant

    if shared.opts.diffusers_pipeline == 'Custom Diffusers Pipeline':
        diffusers_load_config['custom_pipeline'] = shared.opts.custom_diffusers_pipeline

    if shared.opts.data.get('sd_model_checkpoint', '') == 'model.ckpt' or shared.opts.data.get('sd_model_checkpoint', '') == '':
        shared.opts.data['sd_model_checkpoint'] = "runwayml/stable-diffusion-v1-5"

    if op == 'model' or op == 'dict':
        if (model_data.sd_model is not None) and (checkpoint_info is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (checkpoint_info is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return

    sd_model = None

    try:
        if shared.cmd_opts.ckpt is not None and model_data.initial: # initial load
            ckpt_basename = os.path.basename(shared.cmd_opts.ckpt)
            model_name = modelloader.find_diffuser(ckpt_basename)
            if model_name is not None:
                shared.log.info(f'Loading model {op}: {model_name}')
                model_file = modelloader.download_diffusers_model(hub_id=model_name)
                try:
                    shared.log.debug(f'Model load {op} config: {diffusers_load_config}')
                    sd_model = diffusers.DiffusionPipeline.from_pretrained(model_file, **diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Failed loading model: {model_file} {e}')
                list_models() # rescan for downloaded model
                checkpoint_info = CheckpointInfo(model_name)

        checkpoint_info = checkpoint_info or select_checkpoint(op=op)
        if checkpoint_info is None:
            unload_model_weights(op=op)
            return

        vae = None
        sd_vae.loaded_vae_file = None
        if op == 'model' or op == 'refiner':
            vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
            vae = sd_vae.load_vae_diffusers(checkpoint_info.path, vae_file, vae_source)
            if vae is not None:
                diffusers_load_config["vae"] = vae

        if os.path.isdir(checkpoint_info.path):
            err1 = None
            err2 = None
            err3 = None
            try: # try autopipeline first, best choice but not all pipelines are available
                sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                sd_model.model_type = sd_model.__class__.__name__
            except Exception as e:
                err1 = e
            try: # try diffusion pipeline next second-best choice, works for most non-linked pipelines
                if err1 is not None:
                    sd_model = diffusers.DiffusionPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                    sd_model.model_type = sd_model.__class__.__name__
            except Exception as e:
                err2 = e
            try: # try basic pipeline next just in case
                if err2 is not None:
                    sd_model = diffusers.StableDiffusionPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                    sd_model.model_type = sd_model.__class__.__name__
            except Exception as e:
                err3 = e # ignore last error
            if err3 is not None:
                shared.log.error(f'Failed loading {op}: {checkpoint_info.path} auto={err1} diffusion={err2}')
                return
        elif os.path.isfile(checkpoint_info.path) and checkpoint_info.path.lower().endswith('.safetensors'):
            diffusers_load_config["local_files_only"] = True
            diffusers_load_config["extract_ema"] = shared.opts.diffusers_extract_ema
            pipeline, model_type = detect_pipeline(checkpoint_info.path, op)
            if pipeline is None:
                shared.log.error(f'Diffusers {op} pipeline not initialized: {shared.opts.diffusers_pipeline}')
                return
            try:
                if model_type.startswith('Stable Diffusion'):
                    diffusers_load_config['force_zeros_for_empty_prompt '] = shared.opts.diffusers_force_zeros
                    diffusers_load_config['requires_aesthetics_score'] = shared.opts.diffusers_aesthetics_score
                    diffusers_load_config['config_files'] = {
                        'v1': 'configs/v1-inference.yaml',
                        'v2': 'configs/v2-inference-768-v.yaml',
                        'xl': 'configs/sd_xl_base.yaml',
                        'xl_refiner': 'configs/sd_xl_refiner.yaml',
                    }
                if hasattr(pipeline, 'from_single_file'):
                    diffusers_load_config['use_safetensors'] = True
                    sd_model = pipeline.from_single_file(checkpoint_info.path, **diffusers_load_config)
                elif hasattr(pipeline, 'from_ckpt'):
                    sd_model = pipeline.from_ckpt(checkpoint_info.path, **diffusers_load_config)
                else:
                    shared.log.error(f'Diffusers {op} cannot load safetensor model: {checkpoint_info.path} {shared.opts.diffusers_pipeline}')
                    return
                if sd_model is not None:
                    diffusers_load_config.pop('vae', None)
                    diffusers_load_config.pop('safety_checker', None)
                    diffusers_load_config.pop('requires_safety_checker', None)
                    diffusers_load_config.pop('load_safety_checker', None)
                    diffusers_load_config.pop('config_files', None)
                    diffusers_load_config.pop('local_files_only', None)
                    shared.log.debug(f'Setting {op}: pipeline={sd_model.__class__.__name__} config={diffusers_load_config}') # pylint: disable=protected-access
            except Exception as e:
                shared.log.error(f'Diffusers failed loading: {op}={checkpoint_info.path} pipeline={shared.opts.diffusers_pipeline}/{sd_model.__class__.__name__} {e}')
                return
        else:
            shared.log.error(f'Diffusers cannot load: {op}={checkpoint_info.path}')
            return

        if "StableDiffusion" in sd_model.__class__.__name__:
            pass # scheduler is created on first use
        elif "Kandinsky" in sd_model.__class__.__name__:
            sd_model.scheduler.name = 'DDIM'

        set_diffuser_options(sd_model, vae, op)

        base_sent_to_cpu=False
        if (shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none') or shared.opts.ipex_optimize:
            if op == 'refiner' and not getattr(sd_model, 'has_accelerate', False):
                gpu_vram = memory_stats().get('gpu', {})
                free_vram = gpu_vram.get('total', 0) - gpu_vram.get('used', 0)
                refiner_enough_vram = free_vram >= 7 if "StableDiffusionXL" in sd_model.__class__.__name__ else 3
                if not shared.opts.diffusers_move_base and refiner_enough_vram:
                    sd_model.to(devices.device)
                    base_sent_to_cpu=False
                else:
                    if not refiner_enough_vram and not (shared.opts.diffusers_move_base and shared.opts.diffusers_move_refiner):
                        shared.log.warning(f"Insufficient GPU memory, using system memory as fallback: free={free_vram} GB")
                        if not shared.opts.shared.opts.diffusers_seq_cpu_offload and not shared.opts.diffusers_model_cpu_offload:
                            shared.log.debug('Enabled moving base model to CPU')
                            shared.log.debug('Enabled moving refiner model to CPU')
                            shared.opts.diffusers_move_base=True
                            shared.opts.diffusers_move_refiner=True
                    shared.log.debug('Moving base model to CPU')
                    if model_data.sd_model is not None:
                        model_data.sd_model.to(devices.cpu)
                    devices.torch_gc(force=True)
                    sd_model.to(devices.device)
                    base_sent_to_cpu=True
            elif not getattr(sd_model, 'has_accelerate', False):
                sd_model.to(devices.device)

            compile_diffusers(sd_model)

        if sd_model is None:
            shared.log.error('Diffuser model not loaded')
            return
        sd_model.sd_model_hash = checkpoint_info.calculate_shorthash() # pylint: disable=attribute-defined-outside-init
        sd_model.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
        sd_model.sd_model_checkpoint = checkpoint_info.filename # pylint: disable=attribute-defined-outside-init
        shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
        if hasattr(sd_model, "set_progress_bar_config"):
            sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining}', ncols=80, colour='#327fba')
        if op == 'refiner' and shared.opts.diffusers_move_refiner and not getattr(sd_model, 'has_accelerate', False):
            shared.log.debug('Moving refiner model to CPU')
            sd_model.to(devices.cpu)
        elif not getattr(sd_model, 'has_accelerate', False): # In offload modes, accelerate will move models around
            sd_model.to(devices.device)
        if op == 'refiner' and base_sent_to_cpu:
            shared.log.debug('Moving base model back to GPU')
            model_data.sd_model.to(devices.device)
    except Exception as e:
        shared.log.error("Failed to load diffusers model")
        errors.display(e, "loading Diffusers model")

    from modules.textual_inversion import textual_inversion
    sd_model.embedding_db = textual_inversion.EmbeddingDatabase()
    if op == 'refiner':
        model_data.sd_refiner = sd_model
    else:
        model_data.sd_model = sd_model
    sd_model.embedding_db.add_embedding_dir(shared.opts.embeddings_dir)
    sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    timer.record("load")
    devices.torch_gc(force=True)
    script_callbacks.model_loaded_callback(sd_model)
    shared.log.info(f"Loaded {op}: time={timer.summary()} native={get_native(sd_model)} {memory_stats()}")


class DiffusersTaskType(Enum):
    TEXT_2_IMAGE = 1
    IMAGE_2_IMAGE = 2
    INPAINTING = 3
    INSTRUCT = 4


def get_diffusers_task(pipe: diffusers.DiffusionPipeline) -> DiffusersTaskType:
    if pipe.__class__.__name__ == "StableDiffusionXLInstructPix2PixPipeline":
        return DiffusersTaskType.INSTRUCT
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.values():
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.values():
        return DiffusersTaskType.INPAINTING
    else:
        return DiffusersTaskType.TEXT_2_IMAGE


def set_diffuser_pipe(pipe, new_pipe_type):
    sd_checkpoint_info = getattr(pipe, "sd_checkpoint_info", None)
    sd_model_checkpoint = getattr(pipe, "sd_model_checkpoint", None)
    sd_model_hash = getattr(pipe, "sd_model_hash", None)
    has_accelerate = getattr(pipe, "has_accelerate", None)
    embedding_db = getattr(pipe, "embedding_db", None)

    if shared.opts.diffusers_force_inpaint:
        if new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
            new_pipe_type = DiffusersTaskType.INPAINTING # sdxl may work better with init mask
    try:
        if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
            new_pipe = diffusers.AutoPipelineForText2Image.from_pipe(pipe)
        elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
            new_pipe = diffusers.AutoPipelineForImage2Image.from_pipe(pipe)
        elif new_pipe_type == DiffusersTaskType.INPAINTING:
            new_pipe = diffusers.AutoPipelineForInpainting.from_pipe(pipe)
    except Exception: # pylint: disable=unused-variable
        # shared.log.error(f'Failed to change: type={new_pipe_type} pipeline={pipe.__class__.__name__} {e}')
        return pipe

    if pipe.__class__ == new_pipe.__class__:
        return pipe
    new_pipe.sd_checkpoint_info = sd_checkpoint_info
    new_pipe.sd_model_checkpoint = sd_model_checkpoint
    new_pipe.sd_model_hash = sd_model_hash
    new_pipe.has_accelerate = has_accelerate
    new_pipe.embedding_db = embedding_db
    shared.log.debug(f"Pipeline class change: original={pipe.__class__.__name__} target={new_pipe.__class__.__name__}")
    pipe = new_pipe
    return pipe


def get_native(pipe: diffusers.DiffusionPipeline):
    if hasattr(pipe, "vae") and hasattr(pipe.vae.config, "sample_size"):
        # Stable Diffusion
        size = pipe.vae.config.sample_size
    elif hasattr(pipe, "movq") and hasattr(pipe.movq.config, "sample_size"):
        # Kandinsky
        size = pipe.movq.config.sample_size
    elif hasattr(pipe, "unet") and hasattr(pipe.unet.config, "sample_size"):
        size = pipe.unet.config.sample_size
    else:
        size = 0
    return size


def load_model(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'):
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint(op=op)
    if checkpoint_info is None:
        return
    if op == 'model' or op == 'dict':
        if model_data.sd_model is not None and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if model_data.sd_refiner is not None and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return
    shared.log.debug(f'Load {op}: name={checkpoint_info.filename} dict={already_loaded_state_dict is not None}')
    if timer is None:
        timer = Timer()
    current_checkpoint_info = None
    if op == 'model' or op == 'dict':
        if model_data.sd_model is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            current_checkpoint_info = model_data.sd_model.sd_checkpoint_info
            unload_model_weights(op=op)
    else:
        if model_data.sd_refiner is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            current_checkpoint_info = model_data.sd_refiner.sd_checkpoint_info
            unload_model_weights(op=op)

    do_inpainting_hijack()
    devices.set_cuda_params()
    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    if state_dict is None or checkpoint_config is None:
        shared.log.error(f"Failed to load checkpooint: {checkpoint_info.filename}")
        if current_checkpoint_info is not None:
            shared.log.info(f"Restoring previous checkpoint: {current_checkpoint_info.filename}")
            load_model(current_checkpoint_info, None)
        return
    shared.log.debug(f'Model dict loaded: {memory_stats()}')
    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config)
    timer.record("config")
    shared.log.debug(f'Model config loaded: {memory_stats()}')
    sd_model = None
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            clip_is_included_into_sd = sd1_clip_weight in state_dict or sd2_clip_weight in state_dict
            with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd):
                sd_model = instantiate_from_config(sd_config.model)
        except Exception:
            sd_model = instantiate_from_config(sd_config.model)
    for line in stdout.getvalue().splitlines():
        if len(line) > 0:
            shared.log.info(f'LDM: {line.strip()}')
    shared.log.debug(f"Model created from config: {checkpoint_config}")
    sd_model.used_config = checkpoint_config
    sd_model.has_accelerate = False
    timer.record("create")
    ok = load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    if not ok:
        model_data.sd_model = sd_model
        current_checkpoint_info = None
        unload_model_weights(op=op)
        shared.log.debug(f'Model weights unloaded: {memory_stats()} op={op}')
        if op == 'refiner':
            # shared.opts.data['sd_model_refiner'] = 'None'
            shared.opts.sd_model_refiner = 'None'
        return
    else:
        shared.log.debug(f'Model weights loaded: {memory_stats()}')
    timer.record("load")
    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        sd_model.to(devices.device)
    timer.record("move")
    shared.log.debug(f'Model weights moved: {memory_stats()}')
    sd_hijack.model_hijack.hijack(sd_model)
    timer.record("hijack")
    sd_model.eval()
    if op == 'refiner':
        model_data.sd_refiner = sd_model
    else:
        model_data.sd_model = sd_model
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model
    timer.record("embeddings")
    script_callbacks.model_loaded_callback(sd_model)
    timer.record("callbacks")
    shared.log.info(f"Model loaded in {timer.summary()}")
    current_checkpoint_info = None
    devices.torch_gc(force=True)
    shared.log.info(f'Model load finished: {memory_stats()} cached={len(checkpoints_loaded.keys())}')


def reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    load_dict = shared.opts.sd_model_dict != model_data.sd_dict
    from modules import lowvram, sd_hijack
    checkpoint_info = info or select_checkpoint(op=op) # are we selecting model or dictionary
    next_checkpoint_info = info or select_checkpoint(op='dict' if load_dict else 'model') if load_dict else None
    if checkpoint_info is None:
        unload_model_weights(op=op)
        return None
    orig_state = copy.deepcopy(shared.state)
    shared.state = shared_state.State()
    shared.state.begin(f'load-{op}')
    if load_dict:
        shared.log.debug(f'Model dict: existing={sd_model is not None} target={checkpoint_info.filename} info={info}')
    else:
        model_data.sd_dict = 'None'
        shared.log.debug(f'Load model weights: existing={sd_model is not None} target={checkpoint_info.filename} info={info}')
    if sd_model is None:
        sd_model = model_data.sd_model if op == 'model' or op == 'dict' else model_data.sd_refiner
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = getattr(sd_model, 'sd_checkpoint_info', None)
        if current_checkpoint_info is not None and checkpoint_info is not None and current_checkpoint_info.filename == checkpoint_info.filename:
            return None
        if not getattr(sd_model, 'has_accelerate', False):
            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()
            else:
                sd_model.to(devices.cpu)
        if (reuse_dict or shared.opts.model_reuse_dict) and not getattr(sd_model, 'has_accelerate', False):
            shared.log.info('Reusing previous model dictionary')
            sd_hijack.model_hijack.undo_hijack(sd_model)
        else:
            unload_model_weights(op=op)
            sd_model = None
    timer = Timer()
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    timer.record("config")
    if sd_model is None or checkpoint_config != getattr(sd_model, 'used_config', None):
        sd_model = None
        if shared.backend == shared.Backend.ORIGINAL:
            load_model(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
            model_data.sd_dict = shared.opts.sd_model_dict
        else:
            load_diffuser(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
        if load_dict and next_checkpoint_info is not None:
            model_data.sd_dict = shared.opts.sd_model_dict
            shared.opts.data["sd_model_checkpoint"] = next_checkpoint_info.title
            reload_model_weights(reuse_dict=True) # ok we loaded dict now lets redo and load model on top of it
        shared.state.end()
        shared.state = orig_state
        # data['sd_model_checkpoint']
        if op == 'model' or op == 'dict':
            shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
            return model_data.sd_model
        else:
            shared.opts.data["sd_model_refiner"] = checkpoint_info.title
            return model_data.sd_refiner

    # fallback
    shared.log.info(f"Loading using fallback: {op} model={checkpoint_info.title}")
    try:
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        shared.log.error("Load model failed: restoring previous")
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
    finally:
        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")
        script_callbacks.model_loaded_callback(sd_model)
        timer.record("callbacks")
        if sd_model is not None and not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram and not getattr(sd_model, 'has_accelerate', False):
            sd_model.to(devices.device)
            timer.record("device")
    shared.state.end()
    shared.state = orig_state
    shared.log.info(f"Loaded: {op} time={timer.summary()}")
    return sd_model


def disable_offload(sd_model):
    from accelerate.hooks import remove_hook_from_module
    if not getattr(sd_model, 'has_accelerate', False):
        return
    for _name, model in sd_model.components.items():
        if not isinstance(model, torch.nn.Module):
            continue
        remove_hook_from_module(model, recurse=True)


def unload_model_weights(op='model', change_from='none'):
    if op == 'model' or op == 'dict':
        if model_data.sd_model:
            if (shared.backend == shared.Backend.ORIGINAL and change_from != shared.Backend.DIFFUSERS) or change_from == shared.Backend.ORIGINAL:
                from modules import sd_hijack
                model_data.sd_model.to(devices.cpu)
                sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            else:
                disable_offload(model_data.sd_model)
                model_data.sd_model.to('meta')
            model_data.sd_model = None
            shared.log.debug(f'Unload weights {op}: {memory_stats()}')
    else:
        if model_data.sd_refiner:
            if (shared.backend == shared.Backend.ORIGINAL and change_from != shared.Backend.DIFFUSERS) or change_from == shared.Backend.ORIGINAL:
                from modules import sd_hijack
                model_data.sd_model.to(devices.cpu)
                sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            else:
                disable_offload(model_data.sd_model)
                model_data.sd_refiner.to('meta')
            model_data.sd_refiner = None
            shared.log.debug(f'Unload weights {op}: {memory_stats()}')
    devices.torch_gc(force=True)


def apply_token_merging(sd_model, token_merging_ratio=0):
    current_token_merging_ratio = getattr(sd_model, 'applied_token_merged_ratio', 0)
    if token_merging_ratio is None or current_token_merging_ratio is None or current_token_merging_ratio == token_merging_ratio:
        return
    try:
        if current_token_merging_ratio > 0:
            tomesd.remove_patch(sd_model)
    except Exception:
        pass
    if token_merging_ratio > 0:
        if shared.opts.hypertile_unet_enabled and not shared.cmd_opts.experimental:
            shared.log.warning('Token merging not supported with HyperTile for UNet')
            return
        try:
            tomesd.apply_patch(
                sd_model,
                ratio=token_merging_ratio,
                use_rand=False,  # can cause issues with some samplers
                merge_attn=True,
                merge_crossattn=False,
                merge_mlp=False
            )
            shared.log.info(f'Applying token merging: ratio={token_merging_ratio}')
            sd_model.applied_token_merged_ratio = token_merging_ratio
        except Exception:
            shared.log.warning(f'Token merging not supported: pipeline={sd_model.__class__.__name__}')
    else:
        sd_model.applied_token_merged_ratio = 0
