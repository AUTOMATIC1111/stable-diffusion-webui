import collections
import os.path
import re
import io
import threading
from os import mkdir
from urllib import request
from rich import progress # pylint: disable=redefined-builtin
import torch
import safetensors.torch
from omegaconf import OmegaConf
import tomesd
from transformers import logging as transformers_logging
import ldm.modules.midas as midas
from ldm.util import instantiate_from_config
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


class CheckpointInfo:
    def __init__(self, filename):
        name = ''
        self.name = None
        self.hash = None
        self.filename = filename
        abspath = os.path.abspath(filename)
        if shared.backend == shared.Backend.ORIGINAL:
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
        else: # TODO Diffusers
            repo = [r for r in modelloader.diffuser_repos if filename == r['filename']]
            if len(repo) == 0:
                shared.log.error(f'Cannot find diffuser model: {filename}')
                return
            self.name = repo[0]['name']
            self.hash = repo[0]['hash'][:8]
            self.sha256 = repo[0]['hash']
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
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
    if shared.backend == shared.Backend.ORIGINAL:
        model_list = modelloader.load_models(model_path=os.path.join(models_path, 'Stable-diffusion'), model_url=None, command_path=shared.opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
    else:
        model_list = modelloader.load_diffusers(model_path=os.path.join(models_path, 'Diffusers'), command_path=shared.opts.diffusers_dir)
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
                model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
                shared.opts.data['sd_model_checkpoint'] = "v1-5-pruned-emaonly.safetensors"
                model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
                for filename in sorted(model_list, key=str.lower):
                    checkpoint_info = CheckpointInfo(filename)
                    checkpoint_info.register()


def get_closet_checkpoint_match(search_string):
    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info
    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
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
    except:
        return 'NOHASH'


def select_checkpoint():
    model_checkpoint = shared.opts.sd_model_checkpoint
    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        shared.log.debug(f'Select checkpoint: {checkpoint_info.title if checkpoint_info is not None else None}')
        return checkpoint_info
    if len(checkpoints_list) == 0:
        shared.log.error("Cannot run without a checkpoint")
        shared.log.error("Use --ckpt <path-to-checkpoint> to force using existing checkpoint")
        exit(1)
    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        shared.log.warning(f"Selected checkpoint not found: {model_checkpoint}")
        shared.log.warning(f"Loading fallback checkpoint: {checkpoint_info.title}")
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


def read_metadata_from_safetensors(filename):
    import json
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)
        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)
        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass
        return res


def read_state_dict(checkpoint_file, map_location=None): # pylint: disable=unused-argument
    if shared.backend == shared.Backend.DIFFUSERS:
        return None
    try:
        pl_sd = None
        with progress.open(checkpoint_file, 'rb', description=f'Loading weights: [cyan]{checkpoint_file}', auto_refresh=True) as f:
            _, extension = os.path.splitext(checkpoint_file)
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
        # use checkpoint cache
        shared.log.info("Model weights loading: from cache")
        return checkpoints_loaded[checkpoint_info]
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load")
    return res


def load_model_weights(model: torch.nn.Module, checkpoint_info: CheckpointInfo, state_dict, timer):
    shared.log.debug(f'Model weights loading: {memory_stats()}')
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("hash")
    shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    model.load_state_dict(state_dict, strict=False)
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
    devices.dtype_unet = model.model.diffusion_model.dtype
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
    if not "use_ema" in sd_config.model.params:
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


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.initial = True
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.sd_model is None:
            with self.lock:
                try:
                    if shared.backend == shared.Backend.ORIGINAL:
                        load_model()
                    elif shared.backend == shared.Backend.DIFFUSERS:
                        load_diffuser()
                    else:
                        shared.log.error(f"Unknown Stable Diffusion backend: {shared.opts.sd_backend}")
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_model = None
        return self.sd_model

    def set_sd_model(self, v):
        self.sd_model = v


model_data = SdModelData()


def load_diffuser(checkpoint_info=None, already_loaded_state_dict=None, timer=None): # pylint: disable=unused-argument
    if timer is None:
        timer = Timer()
    import diffusers
    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    timer.record("diffusers")
    diffusor_config = {
        "force_download": False,
        "safety_checker": None,
        "resume_download": True,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "cache_dir": shared.opts.diffusers_dir,
        "torch_dtype": devices.dtype,
    }
    if shared.opts.data['sd_model_checkpoint'] == 'model.ckpt':
        shared.opts.data['sd_model_checkpoint'] = "runwayml/stable-diffusion-v1-5"
    sd_model = None
    try:
        if shared.cmd_opts.ckpt is not None and model_data.initial: # initial load
            model_name = modelloader.find_diffuser(shared.cmd_opts.ckpt)
            if model_name is not None:
                shared.log.info(f'Loading diffuser model: {model_name}')
                scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
                sd_model = diffusers.DiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, **diffusor_config)
                list_models() # rescan for downloaded model
                checkpoint_info = CheckpointInfo(model_name)
        if sd_model is None:
            checkpoint_info = checkpoint_info or select_checkpoint()
            shared.log.info(f'Loading diffuser model: {checkpoint_info.filename}')
            scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(checkpoint_info.filename, subfolder="scheduler")
            sd_model = diffusers.DiffusionPipeline.from_pretrained(checkpoint_info.filename, scheduler=scheduler, **diffusor_config)
        if shared.cmd_opts.medvram:
            sd_model.enable_model_cpu_offload()
        if shared.cmd_opts.lowvram:
            sd_model.enable_sequential_cpu_offload()
        if shared.opts.cross_attention_optimization == "xFormers":
            sd_model.enable_xformers_memory_efficient_attention()
        sd_model.sd_checkpoint_info = checkpoint_info
        sd_model.sd_model_checkpoint = checkpoint_info.filename
        sd_model.sd_model_hash = checkpoint_info.hash
        scheduler.name = 'UniPC'
        sd_model.to(devices.device)
    except Exception as e:
        shared.log.error("Failed to load diffusers model")
        errors.display(e, "loading Diffusers model")
    shared.sd_model = sd_model
    timer.record("load")
    shared.log.info(f"Model loaded in {timer.summary()}")
    devices.torch_gc(force=True)
    shared.log.info(f'Model load finished: {memory_stats()}')



def load_model(checkpoint_info=None, already_loaded_state_dict=None, timer=None):
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint()
    if checkpoint_info is None:
        return
    if model_data.sd_model is not None and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
        return
    shared.log.debug(f'Load model: name={checkpoint_info.filename} dict={already_loaded_state_dict is not None}')
    if timer is None:
        timer = Timer()
    current_checkpoint_info = None
    if model_data.sd_model is not None:
        sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
        current_checkpoint_info = model_data.sd_model.sd_checkpoint_info
        unload_model_weights()
        model_data.sd_model = None
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
    shared.log.info(f"Model created from config: {checkpoint_config}")
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
    if shared.cmd_opts.use_ipex and not (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
        sd_model = torch.xpu.optimize(sd_model, dtype=devices.dtype)
        shared.log.info("Applied IPEX Optimize")
    model_data.sd_model = sd_model
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model
    timer.record("embeddings")
    script_callbacks.model_loaded_callback(sd_model)
    timer.record("callbacks")
    shared.log.info(f"Model loaded in {timer.summary()}")
    current_checkpoint_info = None
    devices.torch_gc(force=True)
    shared.log.info(f'Model load finished: {memory_stats()}')


def reload_model_weights(sd_model=None, info=None):
    global skip_next_load # pylint: disable=global-statement
    if skip_next_load:
        shared.log.debug('Reload model weights skip')
        skip_next_load = False
        return
    shared.log.debug(f'Reload model weights: {sd_model is not None} {info}')
    from modules import lowvram, sd_hijack
    checkpoint_info = info or select_checkpoint()
    if not sd_model:
        sd_model = model_data.sd_model
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = sd_model.sd_checkpoint_info
        if checkpoint_info is not None and sd_model.sd_model_checkpoint == checkpoint_info.filename:
            return
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        else:
            sd_model.to(devices.cpu)
    if shared.opts.model_reuse_dict and sd_model is not None:
        shared.log.info('Reusing previous model dictionary')
        sd_hijack.model_hijack.undo_hijack(sd_model)
    else:
        unload_model_weights()
        sd_model = None
    timer = Timer()
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    timer.record("config")
    if sd_model is None or checkpoint_config != sd_model.used_config:
        del sd_model
        checkpoints_loaded.clear()
        if shared.backend == shared.Backend.ORIGINAL:
            load_model(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer)
        else:
            load_diffuser(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer)
        return model_data.sd_model
    try:
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        shared.log.error("Failed to load checkpoint, restoring previous")
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


def unload_model_weights(sd_model=None, _info=None):
    from modules import sd_hijack
    if model_data.sd_model:
        model_data.sd_model.to(devices.cpu)
        if shared.backend == shared.Backend.ORIGINAL:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
        model_data.sd_model = None
        sd_model = None
    devices.torch_gc(force=True)
    shared.log.debug(f'Model weights unloaded: {memory_stats()}')
    return sd_model


def apply_token_merging(sd_model, hr: bool):
    """
    Applies speed and memory optimizations from tomesd.

    Args:
        hr (bool): True if called in the context of a high-res pass
    """

    ratio = shared.opts.token_merging_ratio
    if hr:
        ratio = shared.opts.token_merging_ratio_hr

    tomesd.apply_patch(
        sd_model,
        ratio=ratio,
        max_downsample=shared.opts.token_merging_maximum_down_sampling,
        sx=shared.opts.token_merging_stride_x,
        sy=shared.opts.token_merging_stride_y,
        use_rand=shared.opts.token_merging_random,
        merge_attn=shared.opts.token_merging_merge_attention,
        merge_crossattn=shared.opts.token_merging_merge_cross_attention,
        merge_mlp=shared.opts.token_merging_merge_mlp
    )
