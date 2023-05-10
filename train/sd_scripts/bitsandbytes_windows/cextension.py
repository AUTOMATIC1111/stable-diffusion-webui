import ctypes as ct
from pathlib import Path
from warnings import warn

from .cuda_setup.main import evaluate_cuda_setup


class CUDALibrary_Singleton(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        binary_name = evaluate_cuda_setup()
        package_dir = Path(__file__).parent
        binary_path = package_dir / binary_name

        if not binary_path.exists():
            print(f"CUDA SETUP: TODO: compile library for specific version: {binary_name}")
            legacy_binary_name = "libbitsandbytes.so"
            print(f"CUDA SETUP: Defaulting to {legacy_binary_name}...")
            binary_path = package_dir / legacy_binary_name
            if not binary_path.exists():
                print('CUDA SETUP: CUDA detection failed. Either CUDA driver not installed, CUDA not installed, or you have multiple conflicting CUDA libraries!')
                print('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.')
                raise Exception('CUDA SETUP: Setup Failed!')
            # self.lib = ct.cdll.LoadLibrary(binary_path)
            self.lib = ct.cdll.LoadLibrary(str(binary_path))            # $$$
        else:
            print(f"CUDA SETUP: Loading binary {binary_path}...")
            # self.lib = ct.cdll.LoadLibrary(binary_path)
            self.lib = ct.cdll.LoadLibrary(str(binary_path))            # $$$

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


lib = CUDALibrary_Singleton.get_instance().lib
try:
    lib.cadam32bit_g32
    lib.get_context.restype = ct.c_void_p
    lib.get_cusparse.restype = ct.c_void_p
    COMPILED_WITH_CUDA = True
except AttributeError:
    warn(
        "The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers and GPU quantization are unavailable."
    )
    COMPILED_WITH_CUDA = False
