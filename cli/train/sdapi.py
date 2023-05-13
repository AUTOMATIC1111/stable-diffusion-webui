import asyncio
import aiohttp
import requests
from util import Map


sd_url = "http://127.0.0.1:7860" # automatic1111 api url root
use_session = True
timeout = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
sess = None
quiet = False


async def result(req):
    if req.status != 200:
        if not use_session and sess is not None:
            await sess.close()
        return Map({ 'error': req.status, 'reason': req.reason, 'url': req.url })
    else:
        json = await req.json()
        if type(json) == list:
            res = json
        elif json is None:
            res = {}
        else:
            res = Map(json)
        return res


def resultsync(req: requests.Response):
    if req.status_code != 200:
        return Map({ 'error': req.status_code, 'reason': req.reason, 'url': req.url })
    else:
        json = req.json()
        if type(json) == list:
            res = json
        elif json is None:
            res = {}
        else:
            res = Map(json)
        return res


async def get(endpoint: str, json: dict = None):
    global sess # pylint: disable=global-statement
    sess = sess if sess is not None else await session()
    async with sess.get(url = endpoint, json = json) as req:
        res = await result(req)
        return res


def getsync(endpoint: str, json: dict = None):
    req = requests.get(f'{sd_url}{endpoint}', json = json) # pylint: disable=missing-timeout
    res = resultsync(req)
    return res


async def post(endpoint: str, json: dict = None):
    global sess # pylint: disable=global-statement
    # sess = sess if sess is not None else await session()
    if sess and not sess.closed:
        await sess.close()
    sess = await session()
    async with sess.post(url = endpoint, json = json) as req:
        res = await result(req)
        return res


def postsync(endpoint: str, json: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = json) # pylint: disable=missing-timeout
    res = resultsync(req)
    return res


def interrupt():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    if 'state' in res and res.state.job_count > 0:
        res = postsync('/sdapi/v1/interrupt')
        return res
    else:
        return { 'interrupt': 'idle' }


def progress():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    return res


def options():
    opt = getsync('/sdapi/v1/options')
    flags = getsync('/sdapi/v1/cmd-flags')
    return { 'options': opt, 'flags': flags }


def shutdown():
    postsync('/sdapi/v1/shutdown')


async def session():
    global sess # pylint: disable=global-statement
    time = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
    sess = aiohttp.ClientSession(timeout = time, base_url = sd_url)
    return sess


async def close():
    if sess is not None:
        await asyncio.sleep(0)
        await sess.__aexit__(None, None, None)
