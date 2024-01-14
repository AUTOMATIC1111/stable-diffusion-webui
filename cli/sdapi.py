#!/usr/bin/env python
#pylint: disable=redefined-outer-name
"""
helper methods that creates HTTP session with managed connection pool
provides async HTTP get/post methods and several helper methods
"""

import io
import os
import sys
import ssl
import base64
import asyncio
import logging
import aiohttp
import requests
import urllib3
from PIL import Image
from util import Map, log
from rich import print # pylint: disable=redefined-builtin


sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860") # api url root
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)

use_session = True
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl.create_default_context = ssl._create_unverified_context # pylint: disable=protected-access
timeout = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
sess = None
quiet = False
BaseThreadPolicy = asyncio.WindowsSelectorEventLoopPolicy if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy") else asyncio.DefaultEventLoopPolicy


class AnyThreadEventLoopPolicy(BaseThreadPolicy):
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return super().get_event_loop()
        except (RuntimeError, AssertionError):
            loop = self.new_event_loop()
            self.set_event_loop(loop)
            return loop

asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def authsync():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def auth():
    if sd_username is not None and sd_password is not None:
        return aiohttp.BasicAuth(sd_username, sd_password)
    return None


async def result(req):
    if req.status != 200:
        if not quiet:
            log.error({ 'request error': req.status, 'reason': req.reason, 'url': req.url })
        if not use_session and sess is not None:
            await sess.close()
        return Map({ 'error': req.status, 'reason': req.reason, 'url': req.url })
    else:
        json = await req.json()
        if isinstance(json, list):
            res = json
        elif json is None:
            res = {}
        else:
            res = Map(json)
        log.debug({ 'request': req.status, 'url': req.url, 'reason': req.reason })
        return res


def resultsync(req: requests.Response):
    if req.status_code != 200:
        if not quiet:
            log.error({ 'request error': req.status_code, 'reason': req.reason, 'url': req.url })
        return Map({ 'error': req.status_code, 'reason': req.reason, 'url': req.url })
    else:
        json = req.json()
        if isinstance(json, list):
            res = json
        elif json is None:
            res = {}
        else:
            res = Map(json)
        log.debug({ 'request': req.status_code, 'url': req.url, 'reason': req.reason })
        return res


async def get(endpoint: str, json: dict = None):
    global sess # pylint: disable=global-statement
    sess = sess if sess is not None else await session()
    try:
        async with sess.get(url=endpoint, json=json, verify_ssl=False) as req:
            res = await result(req)
            return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


def getsync(endpoint: str, json: dict = None):
    try:
        req = requests.get(f'{sd_url}{endpoint}', json=json, verify=False, auth=authsync()) # pylint: disable=missing-timeout
        res = resultsync(req)
        return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


async def post(endpoint: str, json: dict = None):
    global sess # pylint: disable=global-statement
    # sess = sess if sess is not None else await session()
    if sess and not sess.closed:
        await sess.close()
    sess = await session()
    try:
        async with sess.post(url=endpoint, json=json, verify_ssl=False) as req:
            res = await result(req)
            return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


def postsync(endpoint: str, json: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json=json, verify=False, auth=authsync()) # pylint: disable=missing-timeout
    res = resultsync(req)
    return res


async def interrupt():
    res = await get('/sdapi/v1/progress?skip_current_image=true')
    if 'state' in res and res.state.job_count > 0:
        log.debug({ 'interrupt': res.state })
        res = await post('/sdapi/v1/interrupt')
        await asyncio.sleep(1)
        return res
    else:
        log.debug({ 'interrupt': 'idle' })
        return { 'interrupt': 'idle' }


def interruptsync():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    if 'state' in res and res.state.job_count > 0:
        log.debug({ 'interrupt': res.state })
        res = postsync('/sdapi/v1/interrupt')
        return res
    else:
        log.debug({ 'interrupt': 'idle' })
        return { 'interrupt': 'idle' }


async def progress():
    res = await get('/sdapi/v1/progress?skip_current_image=false')
    try:
        if res is not None and res.get('current_image', None) is not None:
            res.current_image = Image.open(io.BytesIO(base64.b64decode(res['current_image'])))
    except Exception:
        pass
    log.debug({ 'progress': res })
    return res


def progresssync():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    log.debug({ 'progress': res })
    return res


def get_log():
    res = getsync('/sdapi/v1/log')
    for line in res:
        log.debug(line)
    return res


def get_info():
    import time
    t0 = time.time()
    res = getsync('/sdapi/v1/system-info/status?full=true&refresh=true')
    t1 = time.time()
    print({ 'duration': 1000 * round(t1-t0, 3), **res })
    return res


def options():
    opts = getsync('/sdapi/v1/options')
    flags = getsync('/sdapi/v1/cmd-flags')
    return { 'options': opts, 'flags': flags }


def shutdown():
    try:
        postsync('/sdapi/v1/shutdown')
    except Exception as e:
        log.info({ 'shutdown': e })


async def session():
    global sess # pylint: disable=global-statement
    time = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
    sess = aiohttp.ClientSession(timeout = time, base_url = sd_url, auth=auth())
    log.debug({ 'sdapi': 'session created', 'endpoint': sd_url })
    """
    sess = await aiohttp.ClientSession(timeout = timeout).__aenter__()
    try:
        async with sess.get(url = f'{sd_url}/') as req:
            log.debug({ 'sdapi': 'session created', 'endpoint': sd_url })
    except Exception as e:
        log.error({ 'sdapi': e })
        await asyncio.sleep(0)
        await sess.__aexit__(None, None, None)
        sess = None
    return sess
    """
    return sess


async def close():
    if sess is not None:
        await asyncio.sleep(0)
        await sess.close()
        await sess.__aexit__(None, None, None)
        log.debug({ 'sdapi': 'session closed', 'endpoint': sd_url })


if __name__ == "__main__":
    sys.argv.pop(0)
    log.setLevel(logging.DEBUG)
    if 'interrupt' in sys.argv:
        asyncio.run(interrupt())
    elif 'progress' in sys.argv:
        asyncio.run(progress())
    elif 'progresssync' in sys.argv:
        progresssync()
    elif 'options' in sys.argv:
        opt = options()
        log.debug({ 'options' })
        import json
        print(json.dumps(opt['options'], indent = 2))
        log.debug({ 'cmd-flags' })
        print(json.dumps(opt['flags'], indent = 2))
    elif 'log' in sys.argv:
        get_log()
    elif 'info' in sys.argv:
        get_info()
    elif 'shutdown' in sys.argv:
        shutdown()
    else:
        res = getsync(sys.argv[0])
        print(res)
    asyncio.run(close(), debug=True)
    asyncio.run(asyncio.sleep(0.5))
