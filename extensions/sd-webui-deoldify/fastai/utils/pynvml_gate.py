"""Get OS specific nvml wrapper. On OSX we use pynvx as drop in replacement for pynvml"""

import platform
from ..script import *

#
# BEGIN: Temporary workaround for nvml.dll load issue in Win10
#
# Remove once nicolargo/nvidia-ml-py3#2 and a new version of the module is released 
# (OR fbcotter/py3nvml#10 but will require extra work to rename things)
# Refer https://forums.fast.ai/t/nvml-dll-loading-issue-in-nvidia-ml-py3-7-352-0-py-0/39684/8
import threading
from ctypes import *

nvmlLib = None
libLoadLock = threading.Lock()

def _LoadNvmlLibrary():
    '''
    Load the library if it isn't loaded already
    '''

    global nvmlLib

    if (nvmlLib == None):
        libLoadLock.acquire()

        try:
            if (nvmlLib == None):
                try:
                    if (sys.platform[:3] == "win"):
                        searchPaths = [
                            os.path.join(os.getenv("ProgramFiles", r"C:\Program Files"), r"NVIDIA Corporation\NVSMI\nvml.dll"),
                            os.path.join(os.getenv("WinDir", r"C:\Windows"), r"System32\nvml.dll"),
                        ]
                        nvmlPath = next((x for x in searchPaths if os.path.isfile(x)), None)
                        if (nvmlPath == None):
                            nvmlLib = None
                        else:
                            nvmlLib = CDLL(nvmlPath)
                    else:
                        nvmlLib = None
                except OSError as ose:
                    nvmlLib = None
        finally:
            libLoadLock.release()
#
# END: Temporary workaround for nvml.dll load issue in Win10
#

def load_pynvml_env():
    import pynvml # nvidia-ml-py3

    #
    # BEGIN: Temporary workaround for nvml.dll load issue in Win10 (continued)
    _LoadNvmlLibrary()
    pynvml.nvmlLib = nvmlLib
    #
    # END: Temporary workaround for nvml.dll load issue in Win10
    #

    if platform.system() == "Darwin":
        try:
            from pynvx import pynvml
        except:
            print("please install pynvx on OSX: pip install pynvx")
            sys.exit(1)

        pynvml.nvmlInit()
        return pynvml

    pynvml.nvmlInit()

    return pynvml
