import json
import os
import sys

import platform
import hashlib
import pkg_resources
import psutil
import re

import launch
from modules import paths_internal, timer, shared, extensions, errors

checksum_token = "DontStealMyGamePlz__WINNERS_DONT_USE_DRUGS__DONT_COPY_THAT_FLOPPY"
environment_whitelist = {
    "GIT",
    "INDEX_URL",
    "WEBUI_LAUNCH_LIVE_OUTPUT",
    "GRADIO_ANALYTICS_ENABLED",
    "PYTHONPATH",
    "TORCH_INDEX_URL",
    "TORCH_COMMAND",
    "REQS_FILE",
    "XFORMERS_PACKAGE",
    "CLIP_PACKAGE",
    "OPENCLIP_PACKAGE",
    "STABLE_DIFFUSION_REPO",
    "K_DIFFUSION_REPO",
    "CODEFORMER_REPO",
    "BLIP_REPO",
    "STABLE_DIFFUSION_COMMIT_HASH",
    "K_DIFFUSION_COMMIT_HASH",
    "CODEFORMER_COMMIT_HASH",
    "BLIP_COMMIT_HASH",
    "COMMANDLINE_ARGS",
    "IGNORE_CMD_ARGS_ERRORS",
}


def pretty_bytes(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]:
        if abs(num) < 1024 or unit == 'Y':
            return f"{num:.0f}{unit}{suffix}"
        num /= 1024


def get():
    res = get_dict()

    text = json.dumps(res, ensure_ascii=False, indent=4)

    h = hashlib.sha256(text.encode("utf8"))
    text = text.replace(checksum_token, h.hexdigest())

    return text


re_checksum = re.compile(r'"Checksum": "([0-9a-fA-F]{64})"')


def check(x):
    m = re.search(re_checksum, x)
    if not m:
        return False

    replaced = re.sub(re_checksum, f'"Checksum": "{checksum_token}"', x)

    h = hashlib.sha256(replaced.encode("utf8"))
    return h.hexdigest() == m.group(1)


def get_dict():
    ram = psutil.virtual_memory()

    res = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "Version": launch.git_tag(),
        "Commit": launch.commit_hash(),
        "Script path": paths_internal.script_path,
        "Data path": paths_internal.data_path,
        "Extensions dir": paths_internal.extensions_dir,
        "Checksum": checksum_token,
        "Commandline": get_argv(),
        "Torch env info": get_torch_sysinfo(),
        "Exceptions": errors.get_exceptions(),
        "CPU": {
            "model": platform.processor(),
            "count logical": psutil.cpu_count(logical=True),
            "count physical": psutil.cpu_count(logical=False),
        },
        "RAM": {
            x: pretty_bytes(getattr(ram, x, 0)) for x in ["total", "used", "free", "active", "inactive", "buffers", "cached", "shared"] if getattr(ram, x, 0) != 0
        },
        "Extensions": get_extensions(enabled=True),
        "Inactive extensions": get_extensions(enabled=False),
        "Environment": get_environment(),
        "Config": get_config(),
        "Startup": timer.startup_record,
        "Packages": sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]),
    }

    return res


def get_environment():
    return {k: os.environ[k] for k in sorted(os.environ) if k in environment_whitelist}


def get_argv():
    res = []

    for v in sys.argv:
        if shared.cmd_opts.gradio_auth and shared.cmd_opts.gradio_auth == v:
            res.append("<hidden>")
            continue

        if shared.cmd_opts.api_auth and shared.cmd_opts.api_auth == v:
            res.append("<hidden>")
            continue

        res.append(v)

    return res

re_newline = re.compile(r"\r*\n")


def get_torch_sysinfo():
    try:
        import torch.utils.collect_env
        info = torch.utils.collect_env.get_env_info()._asdict()

        return {k: re.split(re_newline, str(v)) if "\n" in str(v) else v for k, v in info.items()}
    except Exception as e:
        return str(e)


def get_extensions(*, enabled):

    try:
        def to_json(x: extensions.Extension):
            return {
                "name": x.name,
                "path": x.path,
                "version": x.version,
                "branch": x.branch,
                "remote": x.remote,
            }

        return [to_json(x) for x in extensions.extensions if not x.is_builtin and x.enabled == enabled]
    except Exception as e:
        return str(e)


def get_config():
    try:
        return shared.opts.data
    except Exception as e:
        return str(e)
