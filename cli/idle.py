#!/usr/bin/env python

import os
import time
import datetime
import logging
import urllib3
import requests

class Dot(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

opts = Dot({
    "timeout": 3600,
    "frequency": 60,
    "action": "sudo shutdown now",
    "url": "https://127.0.0.1:7860",
    "user": "",
    "password": "",
})

log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level = logging.INFO, format = log_format)
log = logging.getLogger("sd")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
status = None

def progress():
    auth = requests.auth.HTTPBasicAuth(opts.user, opts.password) if opts.user is not None and len(opts.user) > 0 and opts.password is not None and len(opts.password) > 0 else None
    req = requests.get(f'{opts.url}/sdapi/v1/progress?skip_current_image=true', verify=False, auth=auth, timeout=60)
    if req.status_code != 200:
        log.error({ 'url': req.url, 'request': req.status_code, 'reason': req.reason })
        return status
    else:
        res = Dot(req.json())
        log.debug({ 'url': req.url, 'request': req.status_code, 'result': res })
        return res

log.info(f'sdnext monitor started: {opts}')
while True:
    try:
        status = progress()
        state = status.get('state', {})
        last_job = state.get('job_timestamp', None)
        if last_job is None:
            log.warning(f'sdnext montoring cannot get last job info: {status}')
        else:
            last_job = datetime.datetime.strptime(last_job, "%Y%m%d%H%M%S")
            elapsed = datetime.datetime.now() - last_job
            timeout = round(opts.timeout - elapsed.total_seconds())
            log.info(f'sdnext: last_job={last_job} elapsed={elapsed} timeout={timeout}')
            if timeout < 0:
                log.warning(f'sdnext reached: timeout={opts.timeout} action={opts.action}')
                os.system(opts.action)
    except Exception as e:
        log.error(f'sdnext monitor error: {e}')
    finally:
        time.sleep(opts.frequency)
