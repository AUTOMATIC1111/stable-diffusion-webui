import os
import shutil
from urllib.parse import urlparse

from basicsr.utils.download_util import load_file_from_url

from modules.paths import script_path, models_path


def load_models(model_path: str, model_url: str = None, command_path: str = None, dl_name: str = None, existing=None,
                ext_filter=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param dl_name: The file name to use for downloading a model. If not specified, it will be used from the URL.
    @param model_url: If specified, attempt to download model from the given URL.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param existing: An array of existing model paths.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    if ext_filter is None:
        ext_filter = []
    if existing is None:
        existing = []
    try:
        places = []
        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)
        places.append(model_path)
        for place in places:
            if os.path.exists(place):
                for file in os.listdir(place):
                    if os.path.isdir(file):
                        continue
                    if len(ext_filter) != 0:
                        model_name, extension = os.path.splitext(file)
                        if extension not in ext_filter:
                            continue
                    if file not in existing:
                        path = os.path.join(place, file)
                        existing.append(path)
        if model_url is not None and len(existing) == 0:
            if dl_name is not None:
                model_file = load_file_from_url(url=model_url, model_dir=model_path, file_name=dl_name, progress=True)
            else:
                model_file = load_file_from_url(url=model_url, model_dir=model_path, progress=True)

            if os.path.exists(model_file) and os.path.isfile(model_file) and model_file not in existing:
                existing.append(model_file)
    except:
        pass
    return existing


def friendly_name(file: str):
    if "http" in file:
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    model_name = model_name.replace("_", " ").title()
    return model_name


def cleanup_models():
    # This code could probably be more efficient if we used a tuple list or something to store the src/destinations
    # and then enumerate that, but this works for now. In the future, it'd be nice to just have every "model" scaler
    # somehow auto-register and just do these things...
    root_path = script_path
    src_path = models_path
    dest_path = os.path.join(models_path, "Stable-diffusion")
    move_files(src_path, dest_path, ".ckpt")
    src_path = os.path.join(root_path, "ESRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "gfpgan")
    dest_path = os.path.join(models_path, "GFPGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "SwinIR")
    dest_path = os.path.join(models_path, "SwinIR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "repositories/latent-diffusion/experiments/pretrained_models/")
    dest_path = os.path.join(models_path, "LDSR")
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
                    print(f"Moving {file} from {src_path} to {dest_path}.")
                    try:
                        shutil.move(fullpath, dest_path)
                    except:
                        pass
            if len(os.listdir(src_path)) == 0:
                print(f"Removing empty folder: {src_path}")
                shutil.rmtree(src_path, True)
    except:
        pass