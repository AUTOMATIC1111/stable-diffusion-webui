import collections
import os.path
import re
import io
import json
import threading
from os import mkdir
from urllib import request
from enum import Enum
import filelock
from rich import progress # pylint: disable=redefined-builtin
import torch
import safetensors.torch
from omegaconf import OmegaConf
import tomesd
from transformers import logging as transformers_logging
import ldm.modules.midas as midas
from ldm.util import instantiate_from_config
import diffusers
from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config
from modules.sd_hijack_inpainting import do_inpainting_hijack
from modules.timer import Timer
from modules.memstats import memory_stats
from modules.paths_internal import models_path


transformers_logging.set_verbosity_error()
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
checkpoints_list = {}
checkpoint_aliases = {}
checkpoints_loaded = collections.OrderedDict()
skip_next_load = False
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0


class CheckpointInfo:
    def __init__(self, filename):
        name = ''
        self.name = None
        self.hash = None
        self.filename = filename
        self.type = ''
        abspath = os.path.abspath(filename)

        if os.path.isfile(abspath): # ckpt or safetensor
            if shared.opts.ckpt_dir is not None and abspath.startswith(shared.opts.ckpt_dir):
                name = abspath.replace(shared.opts.ckpt_dir, '')
            elif abspath.startswith(model_path):
                name = abspath.replace(model_path, '')
            else:
                name = os.path.basename(filename)
            if name.startswith("\\") or name.startswith("/"):
                name = name[1:]
            self.name = name
            self.hash = model_hash(self.filename)
            self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
            self.path = abspath
            self.type = abspath.split('.')[-1].lower()
            self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
            self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        else: # maybe a diffuser
            repo = [r for r in modelloader.diffuser_repos if filename == r['filename']]
            if len(repo) == 0:
                error_message = f'Cannot find diffuser model: {filename}'
                shared.log.error(error_message)
                raise ValueError(error_message)
            self.name = repo[0]['name']
            self.hash = repo[0]['hash'][:8]
            self.sha256 = repo[0]['hash']
            self.path = repo[0]['path']
            self.type = 'diffusers'
            self.name_for_extra = repo[0]['name']
            self.model_name = repo[0]['name']
            if os.path.isfile(repo[0]['model_info']):
                file_path = repo[0]['model_info']
                with open(file_path, "r", encoding="utf-8") as json_file:
                    try:
                        self.model_info = json.load(json_file)
                    except Exception as e:
                        shared.log.error(f'Error loading model info: {json_file} {e}')
                        self.model_info = {}

        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.title = self.name if self.shorthash is None else f'{self.name} [{self.shorthash}]'
        self.ids = [self.hash, self.model_name, self.title, self.name, f'{self.name} [{self.hash}]'] + ([self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])
        self.metadata = {}
        _, ext = os.path.splitext(self.filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = read_metadata_from_safetensors(filename)
            except Exception as e:
                errors.display(e, f"reading checkpoint metadata: {filename}")

    def register(self):
        checkpoints_list[self.title] = self
        for i in self.ids:
            checkpoint_aliases[i] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return
        self.shorthash = self.sha256[0:10]
        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']
        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()
        return self.shorthash


def setup_model():
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    list_models()
    enable_midas_autodownload()


def checkpoint_tiles():
    def convert(name):
        return int(name) if name.isdigit() else name.lower()
    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()
    ext_filter=[".safetensors"] if shared.opts.sd_disable_ckpt else [".ckpt", ".safetensors"]
    model_list = []
    if shared.backend == shared.Backend.ORIGINAL or shared.opts.diffusers_allow_safetensors:
        model_list += modelloader.load_models(model_path=model_path, model_url=None, command_path=shared.opts.ckpt_dir, ext_filter=ext_filter, download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
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
    shared.log.info(f'Available models: {shared.opts.ckpt_dir} {len(checkpoints_list)}')

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
        ckpt.shorthash = ckpt.sha256[0:10]
        txt.append(f'Calculated full hash: <b>{ckpt.title}</b> {ckpt.shorthash}')
    txt.append(f'Updated full hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
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
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
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
        shared.log.debug(f'Select checkpoint: {op} {checkpoint_info.title if checkpoint_info is not None else None}')
        return checkpoint_info
    if len(checkpoints_list) == 0:
        shared.log.error("Cannot run without a checkpoint")
        shared.log.error("Use --ckpt <path-to-checkpoint> to force using existing checkpoint")
        exit(1)
    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        shared.log.warning(f"Selected checkpoint not found: {model_checkpoint}")
        # shared.log.warning(f"Loading fallback checkpoint: {checkpoint_info.title}")
        shared.opts.data['sd_checkpoint'] = checkpoint_info.title
    shared.log.debug(f'Select checkpoint: {checkpoint_info.title if checkpoint_info is not None else None}')
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
    def default(obj):
        shared.log.debug(f"Model metadata not a valid object: {obj}")
        return str(obj)

    global sd_metadata_pending # pylint: disable=global-statement
    if sd_metadata_pending == 0:
        shared.log.debug(f"Model metadata: {sd_metadata_file} no changes")
        return
    with filelock.FileLock(f"{sd_metadata_file}.lock"):
        try:
            with open(sd_metadata_file, "w", encoding="utf8") as file:
                json.dump(sd_metadata, file, indent=4, skipkeys=True, ensure_ascii=True, check_circular=True, allow_nan=True, default=default)
        except Exception as e:
            shared.log.error(f"Model metadata save error: {sd_metadata_file} {e}")
    shared.log.info(f"Model metadata saved: {sd_metadata_file} {sd_metadata_pending}")
    sd_metadata_pending = 0


def read_metadata_from_safetensors(filename):
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        with filelock.FileLock(f"{sd_metadata_file}.lock"):
            if not os.path.isfile(sd_metadata_file):
                sd_metadata = {}
            else:
                try:
                    with open(sd_metadata_file, "r", encoding="utf8") as file:
                        sd_metadata = json.load(file)
                except Exception:
                    sd_metadata = {}
    res = sd_metadata.get(filename, None)
    if res is not None:
        return res
    res = {}
    try:
        with open(filename, mode="rb") as file:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)
            if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
                shared.log.error(f"Not a valid safetensors file: {filename}")
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        sd_metadata[filename] = res
        global sd_metadata_pending # pylint: disable=global-statement
        sd_metadata_pending += 1
    except Exception as e:
        shared.log.error(f"Error reading metadata from: {filename} {e}")
    return res


def read_state_dict(checkpoint_file, map_location=None): # pylint: disable=unused-argument
    if shared.backend == shared.Backend.DIFFUSERS:
        return None
    try:
        pl_sd = None
        with progress.open(checkpoint_file, 'rb', description=f'Loading weights: [cyan]{checkpoint_file}', auto_refresh=True) as f:
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
    if checkpoint_info in checkpoints_loaded:
        shared.log.info("Model weights loading: from cache")
        return checkpoints_loaded[checkpoint_info]
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load")
    return res


def load_model_weights(model: torch.nn.Module, checkpoint_info: CheckpointInfo, state_dict, timer):
    shared.log.debug(f'Model weights loading: {memory_stats()}')
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("hash")
    if model_data.sd_dict == 'None':
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        shared.log.error(f'Error loading model weights: {checkpoint_info.filename} {e}')
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
    # devices.dtype_unet = model.model.diffusion_model.dtype
    model.model.diffusion_model.to(devices.dtype_unet)
    model.first_stage_model.to(devices.dtype_vae)
    # clean up cache if limit is reached
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)
    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
    model.logvar = model.logvar.to(devices.device)  # fix for training
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("vae")


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
        sd_config.model.params.unet_config.params.use_fp16 = True
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
        if self.sd_model is None:
            with self.lock:
                try:
                    if shared.backend == shared.Backend.ORIGINAL:
                        reload_model_weights(op='model')
                    elif shared.backend == shared.Backend.DIFFUSERS:
                        load_diffuser(op='model')
                    else:
                        shared.log.error(f"Unknown Stable Diffusion backend: {shared.backend}")
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_model = None
        return self.sd_model

    def set_sd_model(self, v):
        self.sd_model = v

    def get_sd_refiner(self):
        if self.sd_model is None:
            with self.lock:
                try:
                    if shared.backend == shared.Backend.ORIGINAL:
                        reload_model_weights(op='refiner')
                    elif shared.backend == shared.Backend.DIFFUSERS:
                        load_diffuser(op='refiner')
                    else:
                        shared.log.error(f"Unknown Stable Diffusion backend: {shared.backend}")
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_refiner = None
        return self.sd_refiner

    def set_sd_refiner(self, v):
        shared.log.debug(f"Class refiner: {v}")
        self.sd_refiner = v

model_data = ModelData()

class PriorPipeline:
    def __init__(self, prior, main):
        self.prior = prior
        self.main = main
        self.scheduler = main.scheduler
        self.tokenizer = self.prior.tokenizer

    def to(self, *args, **kwargs):
        # only the prior is moved to CUDA in a first step
        self.prior.to(*args, **kwargs)

    def enable_model_cpu_offload(self, *args, **kwargs):
        if hasattr(self.prior, 'enable_model_cpu_offload'):
            self.prior.enable_model_cpu_offload(*args, **kwargs)
        self.main.enable_model_cpu_offload(*args, **kwargs)

    def enable_sequential_cpu_offload(self, *args, **kwargs):
        if hasattr(self.prior, 'enable_sequential_cpu_offload'):
            self.prior.enable_sequential_cpu_offload(*args, **kwargs)
        self.main.enable_sequential_cpu_offload(*args, **kwargs)

    def enable_xformers_memory_efficient_attention(self, *args, **kwargs):
        if hasattr(self.prior, 'enable_xformers_memory_efficient_attention'):
            self.prior.enable_xformers_memory_efficient_attention(*args, **kwargs)
        self.main.enable_xformers_memory_efficient_attention(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        unclip_outputs = self.prior(prompt=kwargs.get("prompt"), negative_prompt=kwargs.get("negative_prompt"))

        if self.prior.device.type == "cuda" or self.prior.device.type == "xpu" or self.prior.device.type == "mps":
            prior_device = self.prior.device
            self.prior.to("cpu")
            self.main.to(prior_device)

        kwargs = {**kwargs, **unclip_outputs}
        result = self.main(*args, **kwargs)

        if self.main.device.type == "cuda" or self.main.device.type == "xpu" or self.prior.device.type == "mps":
            main_device = self.main.device
            self.main.to("cpu")
            self.prior.to(main_device)

        return result


def change_backend():
    shared.log.info(f'Pipeline changed: {shared.backend}')
    unload_model_weights()
    checkpoints_loaded.clear()
    from modules.sd_samplers import list_samplers
    list_samplers(shared.backend)


def load_diffuser(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'): # pylint: disable=unused-argument
    import torch # pylint: disable=reimported,redefined-outer-name
    if timer is None:
        timer = Timer()
    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    timer.record("diffusers")
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
        "load_safety_checker": False,
        # "use_safetensors": True,  # TODO(PVP) - we can't enable this for all checkpoints just yet
    }
    if devices.dtype == torch.float16:
        diffusers_load_config['variant'] = 'fp16'

    if shared.opts.data.get('sd_model_checkpoint', '') == 'model.ckpt' or shared.opts.data.get('sd_model_checkpoint', '') == '':
        shared.opts.data['sd_model_checkpoint'] = "runwayml/stable-diffusion-v1-5"

    if op == 'model' or op == 'dict':
        if (model_data.sd_model is not None) and (checkpoint_info is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (checkpoint_info is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return

    shared.log.debug(f'Diffusers load config: {diffusers_load_config}')
    sd_model = None

    try:
        devices.set_cuda_params()
        if shared.cmd_opts.ckpt is not None and model_data.initial: # initial load
            model_name = modelloader.find_diffuser(shared.cmd_opts.ckpt)
            if model_name is not None:
                shared.log.info(f'Loading diffuser {op}: {model_name}')
                model_file = modelloader.download_diffusers_model(hub_id=model_name)
                try:
                    sd_model = diffusers.DiffusionPipeline.from_pretrained(model_file, **diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers failed loading model: {model_file} {e}')
                list_models() # rescan for downloaded model
                checkpoint_info = CheckpointInfo(model_name)

        if sd_model is None:
            checkpoint_info = checkpoint_info or select_checkpoint(op=op)
            if checkpoint_info is None:
                unload_model_weights(op=op)
                return
            shared.log.info(f'Loading diffuser {op}: {checkpoint_info.filename}')

            vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
            vae = sd_vae.load_vae_diffusers(None, vae_file, vae_source)
            if vae is not None:
                diffusers_load_config["vae"] = vae

            if not os.path.isfile(checkpoint_info.path):
                try:
                    sd_model = diffusers.DiffusionPipeline.from_pretrained(checkpoint_info.path, **diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers failed loading model: {checkpoint_info.path} {e}')
            else:
                diffusers_load_config["local_files_only "] = True
                diffusers_load_config["extract_ema"] = shared.opts.diffusers_extract_ema
                try:
                    if shared.opts.diffusers_pipeline == shared.pipelines[0]:
                        pipeline = diffusers.StableDiffusionPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[1]:
                        pipeline = diffusers.StableDiffusionXLPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[2]:
                        pipeline = diffusers.KandinskyPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[3]:
                        pipeline = diffusers.KandinskyV22Pipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[4]:
                        pipeline = diffusers.IFPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[5]:
                        pipeline = diffusers.ShapEPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[6]:
                        pipeline = diffusers.StableDiffusionImg2ImgPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[7]:
                        pipeline = diffusers.StableDiffusionXLImg2ImgPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[8]:
                        pipeline = diffusers.KandinskyImg2ImgPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[9]:
                        pipeline = diffusers.KandinskyV22Img2ImgPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[10]:
                        pipeline = diffusers.IFImg2ImgPipeline
                    elif shared.opts.diffusers_pipeline == shared.pipelines[11]:
                        pipeline = diffusers.ShapEImg2ImgPipeline
                    else:
                        shared.log.error(f'Diffusers unknown pipeline: {shared.opts.diffusers_pipeline}')
                except Exception as e:
                    shared.log.error(f'Diffusers failed initializing pipeline: {shared.opts.diffusers_pipeline} {e}')
                    return
                try:
                    if hasattr(pipeline, 'from_single_file'):
                        diffusers_load_config['use_safetensors'] = True
                        sd_model = pipeline.from_single_file(checkpoint_info.path, **diffusers_load_config)
                    elif hasattr(pipeline, 'from_ckpt'):
                        sd_model = pipeline.from_ckpt(checkpoint_info.path, **diffusers_load_config)
                    else:
                        shared.log.error(f'Diffusers cannot load safetensor model: {checkpoint_info.path} {shared.opts.diffusers_pipeline}')
                        return
                    if sd_model is not None:
                        shared.log.debug(f'Diffusers pipeline: {sd_model.__class__.__name__}') # pylint: disable=protected-access
                except Exception as e:
                    shared.log.error(f'Diffusers failed loading model using pipeline: {checkpoint_info.path} {shared.opts.diffusers_pipeline} {e}')
                    return

        if "StableDiffusion" in sd_model.__class__.__name__:
            pass # scheduler is created on first use
        elif "Kandinsky" in sd_model.__class__.__name__:
            sd_model.scheduler.name = 'DDIM'

        # Prior pipelines
        if hasattr(checkpoint_info, 'model_info') and checkpoint_info.model_info is not None and "prior" in checkpoint_info.model_info:
            prior_id = checkpoint_info.model_info["prior"]
            shared.log.info(f"Loading diffuser prior: {checkpoint_info.filename} {prior_id}")
            prior = diffusers.DiffusionPipeline.from_pretrained(prior_id, **diffusers_load_config)
            sd_model = PriorPipeline(prior=prior, main=sd_model) # wrap sd_model

        if hasattr(sd_model, "enable_model_cpu_offload"):
            if shared.cmd_opts.medvram or shared.opts.diffusers_model_cpu_offload:
                shared.log.debug('Diffusers: enable model CPU offload')
                sd_model.enable_model_cpu_offload()
        if hasattr(sd_model, "enable_sequential_cpu_offload"):
            if shared.opts.diffusers_seq_cpu_offload:
                sd_model.enable_sequential_cpu_offload()
                shared.log.debug('Diffusers: enable sequential CPU offload')
        if hasattr(sd_model, "enable_vae_slicing"):
            if shared.cmd_opts.lowvram or shared.opts.diffusers_vae_slicing:
                shared.log.debug('Diffusers: enable VAE slicing')
                sd_model.enable_vae_slicing()
            else:
                sd_model.disable_vae_slicing()
        if hasattr(sd_model, "enable_vae_tiling"):
            if shared.cmd_opts.lowvram or shared.opts.diffusers_vae_tiling:
                shared.log.debug('Diffusers: enable VAE tiling')
                sd_model.enable_vae_tiling()
            else:
                sd_model.disable_vae_tiling()
        if hasattr(sd_model, "enable_attention_slicing"):
            if shared.cmd_opts.lowvram or shared.opts.diffusers_attention_slicing:
                shared.log.debug('Diffusers: enable attention slicing')
                sd_model.enable_attention_slicing()
            else:
                sd_model.disable_attention_slicing()
        if shared.opts.cross_attention_optimization == "xFormers":
            sd_model.enable_xformers_memory_efficient_attention()
        if shared.opts.opt_channelslast:
            shared.log.debug('Diffusers: enable channels last')
            sd_model.unet.to(memory_format=torch.channels_last)

        base_sent_to_cpu=False
        if shared.opts.cuda_compile and torch.cuda.is_available():
            if op == 'refiner':
                gpu_vram = memory_stats().get('gpu', {})
                free_vram = gpu_vram.get('total', 0) - gpu_vram.get('used', 0)
                refiner_enough_vram = free_vram >= 7 if "StableDiffusionXL" in sd_model.__class__.__name__ else 3
                if not shared.opts.diffusers_move_base and refiner_enough_vram:
                    sd_model.to(devices.device)
                    base_sent_to_cpu=False
                else:
                    if not refiner_enough_vram and not (shared.opts.diffusers_move_base and shared.opts.diffusers_move_refiner):
                        shared.log.warning(f"Insufficient GPU memory, using system memory as fallback: free={free_vram} GB")
                        shared.log.debug('Enabled moving base model to CPU')
                        shared.log.debug('Enabled moving refiner model to CPU')
                        shared.opts.diffusers_move_base=True
                        shared.opts.diffusers_move_refiner=True
                    shared.log.debug('Moving base model to CPU')
                    model_data.sd_model.to("cpu")
                    devices.torch_gc(force=True)
                    sd_model.to(devices.device)
                    base_sent_to_cpu=True
            else:
                sd_model.to(devices.device)
            try:
                shared.log.info(f"Compiling pipeline={sd_model.__class__.__name__} shape={8 * sd_model.unet.config.sample_size} mode={shared.opts.cuda_compile_mode}")
                if shared.opts.cuda_compile_mode == 'ipex':
                    sd_model.unet.training = False
                    sd_model.unet = torch.xpu.optimize(sd_model.unet, dtype=devices.dtype_unet, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
                else:
                    import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
                    log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
                    torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
                    torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
                    torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access
                    sd_model.unet = torch.compile(sd_model.unet, mode=shared.opts.cuda_compile_mode, fullgraph=shared.opts.cuda_compile_fullgraph) # pylint: disable=attribute-defined-outside-init
                    sd_model("dummy prompt")
                shared.log.info("Complilation done.")
            except Exception as err:
                shared.log.warning(f"Model compile not supported: {err}")

        if sd_model is None:
            shared.log.error('Diffuser model not loaded')
            return
        sd_model.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
        sd_model.sd_model_checkpoint = checkpoint_info.filename # pylint: disable=attribute-defined-outside-init
        sd_model.sd_model_hash = checkpoint_info.hash # pylint: disable=attribute-defined-outside-init
        if hasattr(sd_model, "set_progress_bar_config"):
            sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining}', ncols=80, colour='#327fba')
        if op == 'refiner' and shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving refiner model to CPU')
            sd_model.to("cpu")
        else:
            sd_model.to(devices.device)
        if op == 'refiner' and base_sent_to_cpu:
            shared.log.debug('Moving base model back to GPU')
            model_data.sd_model.to(devices.device)
    except Exception as e:
        shared.log.error("Failed to load diffusers model")
        errors.display(e, "loading Diffusers model")

    if op == 'refiner':
        model_data.sd_refiner = sd_model
    else:
        model_data.sd_model = sd_model

    from modules.textual_inversion import textual_inversion
    embedding_db = textual_inversion.EmbeddingDatabase()
    embedding_db.add_embedding_dir(shared.opts.embeddings_dir)
    embedding_db.load_textual_inversion_embeddings(force_reload=True)

    timer.record("load")
    shared.log.info(f"Model loaded in {timer.summary()}")
    devices.torch_gc(force=True)
    shared.log.info(f'Model load finished: {memory_stats()}')


class DiffusersTaskType(Enum):
    TEXT_2_IMAGE = 1
    IMAGE_2_IMAGE = 2
    INPAINTING = 3

def set_diffuser_pipe(pipe, new_pipe_type):
    wrapper_pipe = None

    sd_checkpoint_info = pipe.sd_checkpoint_info
    sd_model_checkpoint = pipe.sd_model_checkpoint
    sd_model_hash = pipe.sd_model_hash

    if pipe.__class__ == PriorPipeline:
        wrapper_pipe = pipe
        pipe = pipe.main

    pipe_name = pipe.__class__.__name__
    pipe_name = pipe_name.replace("Img2Img", "").replace("Inpaint", "")
    new_pipe_cls_str = None
    if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
        new_pipe_cls_str = pipe_name
    elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
        tmp_pipe_name = pipe_name.replace("Pipeline", "Img2ImgPipeline")
        if hasattr(diffusers, tmp_pipe_name):
            new_pipe_cls_str = pipe_name.replace("Pipeline", "Img2ImgPipeline")
    elif new_pipe_type == DiffusersTaskType.INPAINTING:
        tmp_pipe_name = pipe_name.replace("Pipeline", "InpaintPipeline")
        if hasattr(diffusers, tmp_pipe_name):
            new_pipe_cls_str = pipe_name.replace("Pipeline", "InpaintPipeline")

    if new_pipe_cls_str is None:
        shared.log.warning(f'Diffusers unknown pipeline: {tmp_pipe_name}')
        new_pipe_cls_str = pipe_name

    new_pipe_cls = getattr(diffusers, new_pipe_cls_str)

    if pipe.__class__ == new_pipe_cls:
        return

    new_pipe = new_pipe_cls(**pipe.components)

    if wrapper_pipe is not None:
        wrapper_pipe.main = new_pipe
        new_pipe = wrapper_pipe

    new_pipe.sd_checkpoint_info = sd_checkpoint_info
    new_pipe.sd_model_checkpoint = sd_model_checkpoint
    new_pipe.sd_model_hash = sd_model_hash

    model_data.sd_model = new_pipe
    shared.log.info(f"Pipeline class changed from {pipe.__class__.__name__} to {new_pipe_cls.__name__}")


def get_diffusers_task(pipe: diffusers.DiffusionPipeline) -> DiffusersTaskType:
    if pipe.__class__ == PriorPipeline:
        pipe = pipe.main

    if "Img2Img" in pipe.__class__.__name__:
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif "Inpaint" in pipe.__class__.__name__:
        return DiffusersTaskType.INPAINTING
    return DiffusersTaskType.TEXT_2_IMAGE


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
    # shared.log.debug(f'Model config: {sd_config.model.get("params", dict())}')
    try:
        clip_is_included_into_sd = sd1_clip_weight in state_dict or sd2_clip_weight in state_dict
        with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd):
            sd_model = instantiate_from_config(sd_config.model)
    except Exception:
        sd_model = instantiate_from_config(sd_config.model)
    shared.log.debug(f"Model created from config: {checkpoint_config}")
    sd_model.used_config = checkpoint_config
    timer.record("create")
    load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    timer.record("load")
    shared.log.debug(f'Model weights loaded: {memory_stats()}')
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
    global skip_next_load # pylint: disable=global-statement
    if skip_next_load:
        shared.log.debug('Load model weights skip')
        skip_next_load = False
        return
    from modules import lowvram, sd_hijack
    checkpoint_info = info or select_checkpoint(op=op) # are we selecting model or dictionary
    next_checkpoint_info = info or select_checkpoint(op='dict' if load_dict else 'model') if load_dict else None
    if checkpoint_info is None:
        unload_model_weights(op=op)
        return
    if load_dict:
        shared.log.debug(f'Model dict: existing={sd_model is not None} target={checkpoint_info.filename} info={info}')
    else:
        model_data.sd_dict = 'None'
        shared.log.debug(f'Load model weights: existing={sd_model is not None} target={checkpoint_info.filename} info={info}')
    if not sd_model:
        sd_model = model_data.sd_model if op == 'model' or op == 'dict' else model_data.sd_refiner
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = getattr(sd_model, 'sd_checkpoint_info', None)
        if current_checkpoint_info is not None and checkpoint_info is not None and current_checkpoint_info.filename == checkpoint_info.filename:
            return
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        else:
            sd_model.to(devices.cpu)
    if reuse_dict or (shared.opts.model_reuse_dict and sd_model is not None):
        shared.log.info('Reusing previous model dictionary')
        sd_hijack.model_hijack.undo_hijack(sd_model)
    else:
        unload_model_weights(op=op)
        sd_model = None
    timer = Timer()
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    timer.record("config")
    if sd_model is None or checkpoint_config != sd_model.used_config:
        del sd_model
        if shared.backend == shared.Backend.ORIGINAL:
            load_model(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
        else:
            load_diffuser(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
        if load_dict and next_checkpoint_info is not None:
            model_data.sd_dict = shared.opts.sd_model_dict
            shared.opts.data["sd_model_checkpoint"] = next_checkpoint_info.title
            reload_model_weights(reuse_dict=True) # ok we loaded dict now lets redo and load model on top of it
        return model_data.sd_model if op == 'model' or op == 'dict' else model_data.sd_refiner
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
        if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
            sd_model.to(devices.device)
            timer.record("device")
    shared.log.info(f"Weights loaded in {timer.summary()}")


def unload_model_weights(op='model'):
    from modules import sd_hijack
    if op == 'model' or op == 'dict':
        if model_data.sd_model:
            model_data.sd_model.to(devices.cpu)
            if shared.backend == shared.Backend.ORIGINAL:
                sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            model_data.sd_model = None
            shared.log.debug(f'Weights unloaded {op}: {memory_stats()}')
    else:
        if model_data.sd_refiner:
            model_data.sd_refiner.to(devices.cpu)
            if shared.backend == shared.Backend.ORIGINAL:
                sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            model_data.sd_refiner = None
            shared.log.debug(f'Weights unloaded {op}: {memory_stats()}')
    devices.torch_gc(force=True)


def apply_token_merging(sd_model, token_merging_ratio):
    current_token_merging_ratio = getattr(sd_model, 'applied_token_merged_ratio', 0)
    # shared.log.debug(f'Appplying token merging: current={current_token_merging_ratio} target={token_merging_ratio}')
    if current_token_merging_ratio == token_merging_ratio:
        return
    if current_token_merging_ratio > 0:
        tomesd.remove_patch(sd_model)

    if sd_model.__class__ == PriorPipeline:
        # token merging is not supported for PriorPipelines currently
        return

    if token_merging_ratio > 0:
        tomesd.apply_patch(
            sd_model,
            ratio=token_merging_ratio,
            use_rand=False,  # can cause issues with some samplers
            merge_attn=True,
            merge_crossattn=False,
            merge_mlp=False
        )
    sd_model.applied_token_merged_ratio = token_merging_ratio
