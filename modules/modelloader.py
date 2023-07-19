import os
import shutil
import importlib
from typing import Dict
from urllib.parse import urlparse

from modules import shared
from modules.upscaler import Upscaler, UpscalerLanczos, UpscalerNearest, UpscalerNone
from modules.paths import script_path, models_path

diffuser_repos = []


def download_diffusers_model(hub_id: str, cache_dir: str = None, download_config: Dict[str, str] = None, token = None, variant = None, revision = None, mirror = None):
    from diffusers import DiffusionPipeline
    import huggingface_hub as hf

    if download_config is None:
        download_config = {
            "force_download": False,
            "resume_download": True,
            "cache_dir": shared.opts.diffusers_dir,
            # "use_auth_token": True,
        }
    if cache_dir is not None:
        download_config["cache_dir"] = cache_dir
    if variant is not None and len(variant) > 0:
        download_config["variant"] = variant
    if revision is not None and len(revision) > 0:
        download_config["revision"] = revision
    if mirror is not None and len(mirror) > 0:
        download_config["mirror"] = mirror
    shared.log.debug(f"Diffusers downloading: {hub_id} {download_config}")
    if token is not None and len(token) > 2:
        shared.log.debug(f"Diffusers authentication: {token}")
        hf.login(token)
    pipeline_dir = DiffusionPipeline.download(hub_id, **download_config)
    try:
        model_info_dict = hf.model_info(hub_id).cardData # pylint: disable=no-member # TODO Diffusers is this real error?
    except Exception:
        model_info_dict = None
    # some checkpoints need to be downloaded as "hidden" as they just serve as pre- or post-pipelines of other pipelines
    if model_info_dict is not None and "prior" in model_info_dict:
        download_dir = DiffusionPipeline.download(model_info_dict["prior"], **download_config)
        model_info_dict["prior"] = download_dir
        # mark prior as hidden
        with open(os.path.join(download_dir, "hidden"), "w", encoding="utf-8") as f:
            f.write("True")
    shared.writefile(model_info_dict, os.path.join(pipeline_dir, "model_info.json"))
    return pipeline_dir


def load_diffusers_models(model_path: str, command_path: str = None):
    import huggingface_hub as hf
    places = []
    places.append(model_path)
    if command_path is not None and command_path != model_path:
        places.append(command_path)
    diffuser_repos.clear()
    output = []
    for place in places:
        if not os.path.isdir(place):
            continue
        try:
            res = hf.scan_cache_dir(cache_dir=place)
            for r in list(res.repos):
                cache_path = os.path.join(r.repo_path, "snapshots", list(r.revisions)[-1].commit_hash)
                diffuser_repos.append({ 'name': r.repo_id, 'filename': r.repo_id, 'path': cache_path, 'size': r.size_on_disk, 'mtime': r.last_modified, 'hash': list(r.revisions)[-1].commit_hash, 'model_info': str(os.path.join(cache_path, "model_info.json")) })
                if not os.path.isfile(os.path.join(cache_path, "hidden")):
                    output.append(str(r.repo_id))
        except Exception as e:
            shared.log.error(f"Error listing diffusers: {place} {e}")
    shared.log.debug(f'Scanning diffusers cache: {model_path} {command_path} {len(output)}')
    return output


def find_diffuser(name: str):
    import huggingface_hub as hf

    if name in diffuser_repos:
        return name
    if shared.cmd_opts.no_download:
        return None
    hf_api = hf.HfApi()
    hf_filter = hf.ModelFilter(
        model_name=name,
        task='text-to-image',
        library=['diffusers'],
    )
    models = list(hf_api.list_models(filter=hf_filter, full=True, limit=20, sort="downloads", direction=-1))
    shared.log.debug(f'Searching diffusers models: {name} {len(models) > 0}')
    if len(models) > 0:
        return models[0].modelId
    return None


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    places = []
    places.append(model_path)
    if command_path is not None and command_path != model_path and os.path.isdir(command_path):
        places.append(command_path)
    output = []
    try:
        for place in places:
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    shared.log.error(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if full_path not in output:
                    output.append(full_path)
        if model_url is not None and len(output) == 0:
            if download_name is not None:
                from basicsr.utils.download_util import load_file_from_url
                dl = load_file_from_url(model_url, places[0], True, download_name)
                output.append(dl)
            else:
                output.append(model_url)
    except Exception as e:
        shared.log.error(f"Error listing models: {places} {e}")
    return output


def friendly_name(file: str):
    if "http" in file:
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, _extension = os.path.splitext(file)
    return model_name


def cleanup_models():
    # This code could probably be more efficient if we used a tuple list or something to store the src/destinations
    # and then enumerate that, but this works for now. In the future, it'd be nice to just have every "model" scaler
    # somehow auto-register and just do these things...
    root_path = script_path
    src_path = models_path
    dest_path = os.path.join(models_path, "Stable-diffusion")
    # move_files(src_path, dest_path, ".ckpt")
    # move_files(src_path, dest_path, ".safetensors")
    src_path = os.path.join(root_path, "ESRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(models_path, "BSRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path, ".pth")
    src_path = os.path.join(root_path, "gfpgan")
    dest_path = os.path.join(models_path, "GFPGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "SwinIR")
    dest_path = os.path.join(models_path, "SwinIR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "repositories/latent-diffusion/experiments/pretrained_models/")
    dest_path = os.path.join(models_path, "LDSR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "ScuNET")
    dest_path = os.path.join(models_path, "ScuNET")
    move_files(src_path, dest_path)


def move_files(src_path: str, dest_path: str, ext_filter: str = None):
    try:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if os.path.exists(src_path):
            for file in os.listdir(src_path):
                fullpath = os.path.join(src_path, file)
                if os.path.isfile(fullpath):
                    if ext_filter is not None:
                        if ext_filter not in file:
                            continue
                    shared.log.warning(f"Moving {file} from {src_path} to {dest_path}.")
                    try:
                        shutil.move(fullpath, dest_path)
                    except Exception:
                        pass
            if len(os.listdir(src_path)) == 0:
                shared.log.info(f"Removing empty folder: {src_path}")
                shutil.rmtree(src_path, True)
    except Exception:
        pass



def load_upscalers():
    # We can only do this 'magic' method to dynamically load upscalers if they are referenced, so we'll try to import any _model.py files before looking in __subclasses__
    modules_dir = os.path.join(shared.script_path, "modules")
    for file in os.listdir(modules_dir):
        if "_model.py" in file:
            model_name = file.replace("_model.py", "")
            full_model = f"modules.{model_name}_model"
            try:
                importlib.import_module(full_model)
            except Exception:
                pass

    datas = []
    commandline_options = vars(shared.cmd_opts)
    # some of upscaler classes will not go away after reloading their modules, and we'll end up with two copies of those classes. The newest copy will always be the last in the list, so we go from end to beginning and ignore duplicates
    used_classes = {}
    for cls in reversed(Upscaler.__subclasses__()):
        classname = str(cls)
        if classname not in used_classes:
            used_classes[classname] = cls

    for cls in reversed(used_classes.values()):
        name = cls.__name__
        cmd_name = f"{name.lower().replace('upscaler', '')}_models_path"
        commandline_model_path = commandline_options.get(cmd_name, None)
        scaler = cls(commandline_model_path)
        scaler.user_path = commandline_model_path
        scaler.model_download_path = commandline_model_path or scaler.model_path
        datas += scaler.scalers

    shared.sd_upscalers = sorted(
        datas,
        # Special case for UpscalerNone keeps it at the beginning of the list.
        key=lambda x: x.name.lower() if not isinstance(x.scaler, (UpscalerNone, UpscalerLanczos, UpscalerNearest)) else ""
    )
