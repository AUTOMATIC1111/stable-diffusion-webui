# Copyright (c) OpenMMLab. All rights reserved.
"""This file holding some environment constant for sharing by other files."""

import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import torch

import annotator.mmpkg.mmcv as mmcv
from .parrots_wrapper import get_build_config


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - MMCV: MMCV version.
            - MMCV Compiler: The GCC version for compiling MMCV ops.
            - MMCV CUDA Compiler: The CUDA version for compiling MMCV ops.
    """
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        from annotator.mmpkg.mmcv.utils.parrots_wrapper import _get_cuda_home
        CUDA_HOME = _get_cuda_home()
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = mmcv.__version__

    try:
        from annotator.mmpkg.mmcv.ops import get_compiler_version, get_compiling_cuda_version
    except ModuleNotFoundError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV CUDA Compiler'] = 'n/a'
    else:
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV CUDA Compiler'] = get_compiling_cuda_version()

    return env_info
