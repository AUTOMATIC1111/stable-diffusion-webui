from typing import Any, Dict
from fastapi import Depends
from modules import shared
from modules.api import models, helpers


def post_shutdown():
    shared.log.info('Shutdown request received')
    import sys
    sys.exit(0)

def get_motd():
    import requests
    motd = ''
    ver = shared.get_version()
    if ver.get('updated', None) is not None:
        motd = f"version <b>{ver['hash']} {ver['updated']}</b> <span style='color: var(--primary-500)'>{ver['url'].split('/')[-1]}</span><br>"
    if shared.opts.motd:
        res = requests.get('https://vladmandic.github.io/automatic/motd', timeout=10)
        if res.status_code == 200:
            msg = (res.text or '').strip()
            shared.log.info(f'MOTD: {msg if len(msg) > 0 else "N/A"}')
            motd += res.text
    return motd

def get_version():
    return shared.get_version()

def get_platform():
    from installer import get_platform as installer_get_platform
    from modules.loader import get_packages as loader_get_packages
    return { **installer_get_platform(), **loader_get_packages() }

def get_log_buffer(req: models.ReqLog = Depends()):
    lines = shared.log.buffer[:req.lines] if req.lines > 0 else shared.log.buffer.copy()
    if req.clear:
        shared.log.buffer.clear()
    return lines

def get_config():
    options = {}
    for k in shared.opts.data.keys():
        if shared.opts.data_labels.get(k) is not None:
            options.update({k: shared.opts.data.get(k, shared.opts.data_labels.get(k).default)})
        else:
            options.update({k: shared.opts.data.get(k, None)})
    if 'sd_lyco' in options:
        del options['sd_lyco']
    if 'sd_lora' in options:
        del options['sd_lora']
    return options

def set_config(req: Dict[str, Any]):
    updated = []
    for k, v in req.items():
        updated.append({ k: shared.opts.set(k, v) })
    shared.opts.save(shared.config_filename)
    return { "updated": updated }

def get_cmd_flags():
    return vars(shared.cmd_opts)

def get_progress(req: models.ReqProgress = Depends()):
    import time
    if shared.state.job_count == 0:
        return models.ResProgress(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)
    shared.state.do_set_current_image()
    current_image = None
    if shared.state.current_image and not req.skip_current_image:
        current_image = helpers.encode_pil_to_base64(shared.state.current_image)
    batch_x = max(shared.state.job_no, 0)
    batch_y = max(shared.state.job_count, 1)
    step_x = max(shared.state.sampling_step, 0)
    step_y = max(shared.state.sampling_steps, 1)
    current = step_y * batch_x + step_x
    total = step_y * batch_y
    progress = current / total if current > 0 and total > 0 else 0
    time_since_start = time.time() - shared.state.time_start
    eta_relative = (time_since_start / progress) - time_since_start if progress > 0 else 0
    res = models.ResProgress(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)
    return res

def post_interrupt():
    shared.state.interrupt()
    return {}

def post_skip():
    shared.state.skip()

def get_memory():
    try:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
        ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
        ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
    except Exception as err:
        ram = { 'error': f'{err}' }
    try:
        import torch
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
            s = dict(torch.cuda.memory_stats(shared.device))
            allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
            reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
            active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
            inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
            warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
            cuda = {
                'system': system,
                'active': active,
                'allocated': allocated,
                'reserved': reserved,
                'inactive': inactive,
                'events': warnings,
            }
        else:
            cuda = { 'error': 'unavailable' }
    except Exception as err:
        cuda = { 'error': f'{err}' }
    return models.ResMemory(ram = ram, cuda = cuda)
