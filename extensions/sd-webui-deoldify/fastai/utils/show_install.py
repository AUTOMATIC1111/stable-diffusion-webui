from ..script import *
from .collect_env import *

# Temporary POC for module-based script
@call_parse
def main(show_nvidia_smi:Param(opt=False, nargs='?', type=bool)=False):
    return show_install(show_nvidia_smi)

