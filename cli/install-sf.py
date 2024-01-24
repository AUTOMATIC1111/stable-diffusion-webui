#!/usr/bin/env python
import os
import re
import sys

torch_supported = ['211', '212']
cuda_supported = ['cu118', 'cu121']
python_supported = ['39', '310', '311']
repo_url = 'https://github.com/chengzeyi/stable-fast'
api_url = 'https://api.github.com/repos/chengzeyi/stable-fast/releases/tags/nightly'
path_url = '/releases/download/nightly'


def install_pip(arg: str):
    import subprocess
    cmd = f'"{sys.executable}" -m pip install -U {arg}'
    print(f'Running: {cmd}')
    result = subprocess.run(cmd, shell=True, check=False, env=os.environ)
    return result.returncode == 0


def get_nightly():
    import requests
    r = requests.get(api_url, timeout=10)
    if r.status_code != 200:
        print('Failed to get nightly version')
        return None
    json = r.json()
    assets = json.get('assets', [])
    if len(assets) == 0:
        print('Failed to get nightly version')
        return None
    asset = assets[0].get('name', '')
    pattern = r"-(.+?)\+"
    match = re.search(pattern, asset)
    if match:
        ver = match.group(1)
        print(f'Nightly version: {ver}')
        return ver
    else:
        print('Failed to get nightly version')
        return None


def install_stable_fast():
    import torch

    python_ver = f'{sys.version_info.major}{sys.version_info.minor}'
    if python_ver not in python_supported:
        raise ValueError(f'StableFast unsupported python: {python_ver} required {python_supported}')
    if sys.platform == 'linux':
        bin_url = 'manylinux2014_x86_64.whl'
    elif sys.platform == 'win32':
        bin_url = 'win_amd64.whl'
    else:
        raise ValueError(f'StableFast unsupported platform: {sys.platform}')

    torch_ver, cuda_ver = torch.__version__.split('+')
    torch_ver = torch_ver.replace('.', '')
    sf_ver = get_nightly()

    if torch_ver not in torch_supported:
        print(f'StableFast unsupported torch: {torch_ver} required {torch_supported}')
        print('Installing from source...')
        url = 'git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast'
    elif cuda_ver not in cuda_supported:
        print(f'StableFast unsupported CUDA: {cuda_ver} required {cuda_supported}')
        print('Installing from source...')
        url = 'git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast'
    elif sf_ver is None:
        print('StableFast cannot determine version')
        print('Installing from source...')
        url = 'git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast'
    else:
        print('Installing wheel...')
        file_url = f'stable_fast-{sf_ver}+torch{torch_ver}{cuda_ver}-cp{python_ver}-cp{python_ver}-{bin_url}'
        url = f'{repo_url}/{path_url}/{file_url}'

    ok = install_pip(url)
    if ok:
        import sfast
        print(f'StableFast installed: {sfast.__version__}')
    else:
        print('StableFast install failed')

if __name__ == '__main__':
    install_stable_fast()
