# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import platform
import subprocess
import sys

import cv2
import paddle
import paddleseg

IS_WINDOWS = sys.platform == 'win32'


def _find_cuda_home():
    '''Finds the CUDA install path. It refers to the implementation of
    pytorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py>.
    '''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            nvcc = subprocess.check_output([which,
                                            'nvcc']).decode().rstrip('\r\n')
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home


def _get_nvcc_info(cuda_home):
    if cuda_home is not None and os.path.isdir(cuda_home):
        try:
            nvcc = os.path.join(cuda_home, 'bin/nvcc')
            if not IS_WINDOWS:
                nvcc = subprocess.check_output(
                    "{} -V".format(nvcc), shell=True).decode()
            else:
                nvcc = subprocess.check_output(
                    "\"{}\" -V".format(nvcc), shell=True).decode()
            nvcc = nvcc.strip().split('\n')[-1]
        except subprocess.SubprocessError:
            nvcc = "Not Available"
    else:
        nvcc = "Not Available"
    return nvcc


def _get_gpu_info():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi',
                                            '-L']).decode().strip()
        gpu_info = gpu_info.split('\n')
        for i in range(len(gpu_info)):
            gpu_info[i] = ' '.join(gpu_info[i].split(' ')[:4])
    except:
        gpu_info = ' Can not get GPU information. Please make sure CUDA have been installed successfully.'
    return gpu_info


def get_sys_env():
    """collect environment information"""
    env_info = {}
    env_info['platform'] = platform.platform()

    env_info['Python'] = sys.version.replace('\n', '')

    # TODO is_compiled_with_cuda() has not been moved
    compiled_with_cuda = paddle.is_compiled_with_cuda()
    env_info['Paddle compiled with cuda'] = compiled_with_cuda

    if compiled_with_cuda:
        cuda_home = _find_cuda_home()
        env_info['NVCC'] = _get_nvcc_info(cuda_home)
        # refer to https://github.com/PaddlePaddle/Paddle/blob/release/2.0-rc/paddle/fluid/platform/device_context.cc#L327
        v = paddle.get_cudnn_version()
        v = str(v // 1000) + '.' + str(v % 1000 // 100)
        env_info['cudnn'] = v
        if 'gpu' in paddle.get_device():
            gpu_nums = paddle.distributed.ParallelEnv().nranks
        else:
            gpu_nums = 0
        env_info['GPUs used'] = gpu_nums

        env_info['CUDA_VISIBLE_DEVICES'] = os.environ.get(
            'CUDA_VISIBLE_DEVICES')
        if gpu_nums == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        env_info['GPU'] = _get_gpu_info()

    try:
        gcc = subprocess.check_output(['gcc', '--version']).decode()
        gcc = gcc.strip().split('\n')[0]
        env_info['GCC'] = gcc
    except:
        pass

    env_info['PaddleSeg'] = paddleseg.__version__
    env_info['PaddlePaddle'] = paddle.__version__
    env_info['OpenCV'] = cv2.__version__

    return env_info
