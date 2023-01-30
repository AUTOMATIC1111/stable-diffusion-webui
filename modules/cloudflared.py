import atexit
import os
import platform
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm

from modules import paths


def _get_command() -> str:
    system = platform.system()
    machine = platform.machine()
    if system == "Windows":
        if machine == "AMD64":
            command = "cloudflared-windows-amd64.exe"
        elif machine == "x86":
            command = "cloudflared-windows-386.exe"
        else:
            raise SystemError(f"{machine} is not supported on Windows")
    elif system == "Linux":
        if machine == "x86_64":
            command = "cloudflared-linux-amd64"
        elif machine == "i386":
            command = "cloudflared-linux-386"
        elif machine == "arm":
            command = "cloudflared-linux-arm"
        elif machine == "arm64":
            command = "cloudflared-linux-arm64"
        elif machine == "aarch":
            command = "cloudflared-linux-arm64"
        else:
            raise SystemError(f"{machine} is not supported on Linux")
    elif system == "Darwin":
        if machine == "x86_64":
            command = "cloudflared"
        elif machine == "arm64":
            print(
                "* On a MacOS system with an Apple Silicon chip, Rosetta 2 needs to be installed, refer to this guide to learn more: https://support.apple.com/en-us/HT211861"
            )
            command = "cloudflared"
        else:
            raise SystemError(f"{machine} is not supported on Darwin")
    else:
        raise SystemError(f"{system} is not supported")
    return command


# Needed for the darwin package
def _extract_tarball(tar_path: str, filename: str):
    tar = tarfile.open(tar_path + "/" + filename, "r")
    for item in tar:
        tar.extract(item, tar_path)


def _run_cloudflared(port: Union[int, str]):
    system = platform.system()
    machine = platform.machine()
    command = _get_command()
    cloudflared_path = os.path.join(paths.data_path, "tmp")
    if not os.path.exists(cloudflared_path):
        os.mkdir(cloudflared_path)

    # Untar on Darwin, as there is an exclusive binary.
    if system == "Darwin":
        _download_cloudflared(cloudflared_path, "cloudflared-darwin-amd64.tgz")
        _extract_tarball(cloudflared_path, "cloudflared-darwin-amd64.tgz")
        executable = str(Path(cloudflared_path, command))
    else:
        _download_cloudflared(cloudflared_path, command)
        executable = str(Path(cloudflared_path, command))
    os.chmod(executable, 0o777)

    if system == "Darwin" and machine == "arm64":
        cloudflared = subprocess.Popen(
            [
                "arch",
                "-x86_64",
                executable,
                "tunnel",
                "--url",
                f"http://127.0.0.1:{port}",
                "--metrics",
                "127.0.0.1:8099",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        cloudflared = subprocess.Popen(
            [
                executable,
                "tunnel",
                "--url",
                f"http://127.0.0.1:{port}",
                "--metrics",
                "127.0.0.1:8099",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    atexit.register(cloudflared.terminate)
    localhost_url = "http://127.0.0.1:8099/metrics"
    attempts = 0
    while attempts < 10:
        try:
            tunnel_url = requests.get(localhost_url).text
            tunnel_url = re.search(r"(?P<url>https?://\S+.trycloudflare.com)", tunnel_url).group("url")
            break
        except Exception:
            attempts += 1
            time.sleep(3)
            continue
    if attempts == 10:
        raise RuntimeError("Can't connect to Cloudflare Edge")
    return tunnel_url


def _download_cloudflared(cloudflared_path: str, command: str):
    system = platform.system()
    machine = platform.machine()
    if Path(cloudflared_path, command).exists():
        if system == "Darwin" and machine == "arm64":
            update_cloudflared = subprocess.Popen(
                ["arch", "-x86_64", cloudflared_path + "/" + "cloudflared", "update"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        elif system == "Darwin" and machine == "x86_64":
            update_cloudflared = subprocess.Popen(
                [cloudflared_path + "/" + "cloudflared", "update"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        else:
            update_cloudflared = subprocess.Popen(
                [cloudflared_path + "/" + command, "update"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        return

    url = None
    if system == "Windows":
        if machine == "AMD64":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        elif machine == "x86":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-386.exe"

    elif system == "Linux":
        if machine == "x86_64":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        elif machine == "i386":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-386"
        elif machine == "arm":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm"
        elif machine == "arm64":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        elif machine == "aarch64":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"

    elif system == "Darwin":
        if machine == "x86_64":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
        if machine == "arm64":
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"

    if url is None:
        raise SystemError(f"cloudflared: {system} {machine} is not supported")
    _download_file(cloudflared_path, url)


def _download_file(cloudflared_path: str, url: str) -> str:
    local_filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    download_path = str(Path(cloudflared_path, local_filename))
    with tqdm.wrapattr(
        r.raw,
        "read",
        total=int(r.headers.get("content-length", 0)),
        desc="Downloading cloudflared...",
        leave=False,
    ) as src:
        with open(download_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return download_path


def start_cloudflared(port: Union[int, str]):
    cloudflared_address = _run_cloudflared(port)
    print(f" * Running on {cloudflared_address}")
    print(" * Traffic stats available on http://127.0.0.1:8099/metrics")
