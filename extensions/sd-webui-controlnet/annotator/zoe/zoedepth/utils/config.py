# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import json
import os

from .easydict import EasyDict as edict
from .arg_utils import infer_type

import pathlib
import platform

ROOT = pathlib.Path(__file__).parent.parent.resolve()

HOME_DIR = os.path.expanduser("~")

COMMON_CONFIG = {
    "save_dir": os.path.expanduser("~/shortcuts/monodepth3_checkpoints"),
    "project": "ZoeDepth",
    "tags": '',
    "notes": "",
    "gpu": None,
    "root": ".",
    "uid": None,
    "print_losses": False
}

DATASETS_CONFIG = {
    "kitti": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/raw"),
        "gt_path": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/gts"),
        "filenames_file": "./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 1216,  # 704
        "data_path_eval": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/raw"),
        "gt_path_eval": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/gts"),
        "filenames_file_eval": "./train_test_inputs/kitti_eigen_test_files_with_gt.txt",

        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,

        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    },
    "kitti_test": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/raw"),
        "gt_path": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/gts"),
        "filenames_file": "./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 1216,
        "data_path_eval": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/raw"),
        "gt_path_eval": os.path.join(HOME_DIR, "shortcuts/datasets/kitti/gts"),
        "filenames_file_eval": "./train_test_inputs/kitti_eigen_test_files_with_gt.txt",

        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,

        "do_random_rotate": False,
        "degree": 1.0,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    },
    "nyu": {
        "dataset": "nyu",
        "avoid_boundary": False,
        "min_depth": 1e-3,   # originally 0.1
        "max_depth": 10,
        "data_path": os.path.join(HOME_DIR, "shortcuts/datasets/nyu_depth_v2/sync/"),
        "gt_path": os.path.join(HOME_DIR, "shortcuts/datasets/nyu_depth_v2/sync/"),
        "filenames_file": "./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        "input_height": 480,
        "input_width": 640,
        "data_path_eval": os.path.join(HOME_DIR, "shortcuts/datasets/nyu_depth_v2/official_splits/test/"),
        "gt_path_eval": os.path.join(HOME_DIR, "shortcuts/datasets/nyu_depth_v2/official_splits/test/"),
        "filenames_file_eval": "./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,
        "min_depth_diff": -10,
        "max_depth_diff": 10,

        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": True
    },
    "ibims": {
        "dataset": "ibims",
        "ibims_root": os.path.join(HOME_DIR, "shortcuts/datasets/ibims/ibims1_core_raw/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "sunrgbd": {
        "dataset": "sunrgbd",
        "sunrgbd_root": os.path.join(HOME_DIR, "shortcuts/datasets/SUNRGBD/test/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 8,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "diml_indoor": {
        "dataset": "diml_indoor",
        "diml_indoor_root": os.path.join(HOME_DIR, "shortcuts/datasets/diml_indoor_test/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "diml_outdoor": {
        "dataset": "diml_outdoor",
        "diml_outdoor_root": os.path.join(HOME_DIR, "shortcuts/datasets/diml_outdoor_test/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": False,
        "min_depth_eval": 2,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80
    },
    "diode_indoor": {
        "dataset": "diode_indoor",
        "diode_indoor_root": os.path.join(HOME_DIR, "shortcuts/datasets/diode_indoor/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "diode_outdoor": {
        "dataset": "diode_outdoor",
        "diode_outdoor_root": os.path.join(HOME_DIR, "shortcuts/datasets/diode_outdoor/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80
    },
    "hypersim_test": {
        "dataset": "hypersim_test",
        "hypersim_test_root": os.path.join(HOME_DIR, "shortcuts/datasets/hypersim_test/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "vkitti": {
        "dataset": "vkitti",
        "vkitti_root": os.path.join(HOME_DIR, "shortcuts/datasets/vkitti_test/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": True,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80
    },
    "vkitti2": {
        "dataset": "vkitti2",
        "vkitti2_root": os.path.join(HOME_DIR, "shortcuts/datasets/vkitti2/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": True,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80,
    },
    "ddad": {
        "dataset": "ddad",
        "ddad_root": os.path.join(HOME_DIR, "shortcuts/datasets/ddad/ddad_val/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": True,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80,
    },
}

ALL_INDOOR = ["nyu", "ibims", "sunrgbd", "diode_indoor", "hypersim_test"]
ALL_OUTDOOR = ["kitti", "diml_outdoor", "diode_outdoor",  "vkitti2", "ddad"]
ALL_EVAL_DATASETS = ALL_INDOOR + ALL_OUTDOOR

COMMON_TRAINING_CONFIG = {
    "dataset": "nyu",
    "distributed": True,
    "workers": 16,
    "clip_grad": 0.1,
    "use_shared_dict": False,
    "shared_dict": None,
    "use_amp": False,

    "aug": True,
    "random_crop": False,
    "random_translate": False,
    "translate_prob": 0.2,
    "max_translation": 100,

    "validate_every": 0.25,
    "log_images_every": 0.1,
    "prefetch": False,
}


def flatten(config, except_keys=('bin_conf')):
    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key in except_keys:
                    yield (key, value)
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield (key, value)

    return dict(list(recurse(config)))


def split_combined_args(kwargs):
    """Splits the arguments that are combined with '__' into multiple arguments.
       Combined arguments should have equal number of keys and values.
       Keys are separated by '__' and Values are separated with ';'.
       For example, '__n_bins__lr=256;0.001'

    Args:
        kwargs (dict): key-value pairs of arguments where key-value is optionally combined according to the above format. 

    Returns:
        dict: Parsed dict with the combined arguments split into individual key-value pairs.
    """
    new_kwargs = dict(kwargs)
    for key, value in kwargs.items():
        if key.startswith("__"):
            keys = key.split("__")[1:]
            values = value.split(";")
            assert len(keys) == len(
                values), f"Combined arguments should have equal number of keys and values. Keys are separated by '__' and Values are separated with ';'. For example, '__n_bins__lr=256;0.001. Given (keys,values) is ({keys}, {values})"
            for k, v in zip(keys, values):
                new_kwargs[k] = v
    return new_kwargs


def parse_list(config, key, dtype=int):
    """Parse a list of values for the key if the value is a string. The values are separated by a comma. 
    Modifies the config in place.
    """
    if key in config:
        if isinstance(config[key], str):
            config[key] = list(map(dtype, config[key].split(',')))
        assert isinstance(config[key], list) and all([isinstance(e, dtype) for e in config[key]]
                                                     ), f"{key} should be a list of values dtype {dtype}. Given {config[key]} of type {type(config[key])} with values of type {[type(e) for e in config[key]]}."


def get_model_config(model_name, model_version=None):
    """Find and parse the .json config file for the model.

    Args:
        model_name (str): name of the model. The config file should be named config_{model_name}[_{model_version}].json under the models/{model_name} directory.
        model_version (str, optional): Specific config version. If specified config_{model_name}_{model_version}.json is searched for and used. Otherwise config_{model_name}.json is used. Defaults to None.

    Returns:
        easydict: the config dictionary for the model.
    """
    config_fname = f"config_{model_name}_{model_version}.json" if model_version is not None else f"config_{model_name}.json"
    config_file = os.path.join(ROOT, "models", model_name, config_fname)
    if not os.path.exists(config_file):
        return None

    with open(config_file, "r") as f:
        config = edict(json.load(f))

    # handle dictionary inheritance
    # only training config is supported for inheritance
    if "inherit" in config.train and config.train.inherit is not None:
        inherit_config = get_model_config(config.train["inherit"]).train
        for key, value in inherit_config.items():
            if key not in config.train:
                config.train[key] = value
    return edict(config)


def update_model_config(config, mode, model_name, model_version=None, strict=False):
    model_config = get_model_config(model_name, model_version)
    if model_config is not None:
        config = {**config, **
                  flatten({**model_config.model, **model_config[mode]})}
    elif strict:
        raise ValueError(f"Config file for model {model_name} not found.")
    return config


def check_choices(name, value, choices):
    # return  # No checks in dev branch
    if value not in choices:
        raise ValueError(f"{name} {value} not in supported choices {choices}")


KEYS_TYPE_BOOL = ["use_amp", "distributed", "use_shared_dict", "same_lr", "aug", "three_phase",
                  "prefetch", "cycle_momentum"]  # Casting is not necessary as their int casted values in config are 0 or 1


def get_config(model_name, mode='train', dataset=None, **overwrite_kwargs):
    """Main entry point to get the config for the model.

    Args:
        model_name (str): name of the desired model.
        mode (str, optional): "train" or "infer". Defaults to 'train'.
        dataset (str, optional): If specified, the corresponding dataset configuration is loaded as well. Defaults to None.
    
    Keyword Args: key-value pairs of arguments to overwrite the default config.

    The order of precedence for overwriting the config is (Higher precedence first):
        # 1. overwrite_kwargs
        # 2. "config_version": Config file version if specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{config_version}.json
        # 3. "version_name": Default Model version specific config specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{version_name}.json
        # 4. common_config: Default config for all models specified in COMMON_CONFIG

    Returns:
        easydict: The config dictionary for the model.
    """


    check_choices("Model", model_name, ["zoedepth", "zoedepth_nk"])
    check_choices("Mode", mode, ["train", "infer", "eval"])
    if mode == "train":
        check_choices("Dataset", dataset, ["nyu", "kitti", "mix", None])

    config = flatten({**COMMON_CONFIG, **COMMON_TRAINING_CONFIG})
    config = update_model_config(config, mode, model_name)

    # update with model version specific config
    version_name = overwrite_kwargs.get("version_name", config["version_name"])
    config = update_model_config(config, mode, model_name, version_name)

    # update with config version if specified
    config_version = overwrite_kwargs.get("config_version", None)
    if config_version is not None:
        print("Overwriting config with config_version", config_version)
        config = update_model_config(config, mode, model_name, config_version)

    # update with overwrite_kwargs
    # Combined args are useful for hyperparameter search
    overwrite_kwargs = split_combined_args(overwrite_kwargs)
    config = {**config, **overwrite_kwargs}

    # Casting to bool   # TODO: Not necessary. Remove and test
    for key in KEYS_TYPE_BOOL:
        if key in config:
            config[key] = bool(config[key])

    # Model specific post processing of config
    parse_list(config, "n_attractors")

    # adjust n_bins for each bin configuration if bin_conf is given and n_bins is passed in overwrite_kwargs
    if 'bin_conf' in config and 'n_bins' in overwrite_kwargs:
        bin_conf = config['bin_conf']  # list of dicts
        n_bins = overwrite_kwargs['n_bins']
        new_bin_conf = []
        for conf in bin_conf:
            conf['n_bins'] = n_bins
            new_bin_conf.append(conf)
        config['bin_conf'] = new_bin_conf

    if mode == "train":
        orig_dataset = dataset
        if dataset == "mix":
            dataset = 'nyu'  # Use nyu as default for mix. Dataset config is changed accordingly while loading the dataloader
        if dataset is not None:
            config['project'] = f"MonoDepth3-{orig_dataset}"  # Set project for wandb

    if dataset is not None:
        config['dataset'] = dataset
        config = {**DATASETS_CONFIG[dataset], **config}
        

    config['model'] = model_name
    typed_config = {k: infer_type(v) for k, v in config.items()}
    # add hostname to config
    config['hostname'] = platform.node()
    return edict(typed_config)


def change_dataset(config, new_dataset):
    config.update(DATASETS_CONFIG[new_dataset])
    return config
