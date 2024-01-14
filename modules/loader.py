from __future__ import annotations
import re
import sys
import logging
import warnings
import urllib3
from modules import timer, errors

initialized = False
errors.install()
logging.getLogger("DeepSpeed").disabled = True
# os.environ.setdefault('OMP_NUM_THREADS', 1)
# os.environ.setdefault('MKL_NUM_THREADS', 1)

# import tensorflow as tf # pylint: disable=C0411

import torch # pylint: disable=C0411

# torch.set_num_threads(1)
try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
    errors.log.debug(f'Load IPEX=={ipex.__version__}')
except Exception:
    pass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
import torchvision # pylint: disable=W0611,C0411
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them # pylint: disable=W0611,C0411
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
logging.getLogger("pytorch_lightning").disabled = True
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)
timer.startup.record("torch")

from fastapi import FastAPI # pylint: disable=W0611,C0411
import gradio # pylint: disable=W0611,C0411
timer.startup.record("gradio")
errors.install([gradio])

import pydantic # pylint: disable=W0611,C0411
timer.startup.record("pydantic")

import diffusers # pylint: disable=W0611,C0411
timer.startup.record("diffusers")
errors.log.info(f'Load packages: torch={getattr(torch, "__long_version__", torch.__version__)} diffusers={diffusers.__version__} gradio={gradio.__version__}')

try:
    import os
    import math
    cores = os.cpu_count()
    affinity = len(os.sched_getaffinity(0))
    threads = torch.get_num_threads()
    if threads < (affinity / 2):
        torch.set_num_threads(math.floor(affinity / 2))
        threads = torch.get_num_threads()
        errors.log.debug(f'Detected: cores={cores} affinity={affinity} set threads={threads}')
except Exception:
    pass

try: # fix changed import in torchvision 0.17+, which breaks basicsr
    import torchvision.transforms.functional_tensor # pylint: disable=unused-import, ungrouped-imports
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...
