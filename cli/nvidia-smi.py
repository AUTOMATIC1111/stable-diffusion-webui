#!/usr/bin/env python
import os
import json
import shutil
import subprocess
import xmltodict
from rich import print # pylint: disable=redefined-builtin
from util import log, Map

def get_nvidia_smi(output='dict'):
    smi = shutil.which('nvidia-smi')
    if smi is None:
        log.error("nvidia-smi not found")
        return None
    result = subprocess.run(f'"{smi}" -q -x', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    xml = result.stdout.decode(encoding="utf8", errors="ignore")
    d = xmltodict.parse(xml)
    if 'nvidia_smi_log' in d:
        d = d['nvidia_smi_log']
    if 'gpu' in d and 'supported_clocks' in d['gpu']:
        del d['gpu']['supported_clocks']
    if output == 'dict':
        return d
    elif output == 'class' or output == 'map':
        d = Map(d)
        return d
    elif output == 'json':
        return json.dumps(d, indent=4)

if __name__ == "__main__":
    res = get_nvidia_smi(output='dict')
    print(type(res), res)
