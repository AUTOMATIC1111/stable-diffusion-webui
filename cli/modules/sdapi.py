#!/bin/env python
"""
helper methods that creates HTTP session with managed connection pool
provides async HTTP get/post methods and several helper methods
"""

import sys
import json
import aiohttp
import asyncio
import logging
import requests
from util import Map, log


sd_url = "http://127.0.0.1:7860" # automatic1111 api url root
use_session = True
timeout = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
sess = None
quiet = False


async def result(req):
    if req.status != 200:
        if not quiet:
            log.error({ 'request error': req.status, 'reason': req.reason, 'url': req.url })
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
        log.debug({ 'request': req.status, 'url': req.url, 'reason': req.reason })
        return res


def resultsync(req: requests.Response):
    if req.status_code != 200:
        if not quiet:
            log.error({ 'request error': req.status_code, 'reason': req.reason, 'url': req.url })
        return Map({ 'error': req.status_code, 'reason': req.reason, 'url': req.url })
    else:
        json = req.json()
        if type(json) == list:
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
        async with sess.get(url = endpoint, json = json) as req:
            res = await result(req)
            return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


def getsync(endpoint: str, json: dict = None):
    try:
        req = requests.get(f'{sd_url}{endpoint}', json = json) # pylint: disable=missing-timeout
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
        async with sess.post(url = endpoint, json = json) as req:
            res = await result(req)
            return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


def postsync(endpoint: str, json: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = json) # pylint: disable=missing-timeout
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
    res = await get('/sdapi/v1/progress?skip_current_image=true')
    log.debug({ 'progress': res })
    return res


def progresssync():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    log.debug({ 'progress': res })
    return res


def options():
    options = getsync('/sdapi/v1/options')
    flags = getsync('/sdapi/v1/cmd-flags')
    return { 'options': options, 'flags': flags }


def shutdown():
    try:
        postsync('/sdapi/v1/shutdown')
    except Exception as e:
        log.info({ 'shutdown': e })


async def session():
    global sess # pylint: disable=global-statement
    time = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
    sess = aiohttp.ClientSession(timeout = time, base_url = sd_url)
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
        await sess.__aexit__(None, None, None)
        log.debug({ 'sdapi': 'session closed', 'endpoint': sd_url })


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    if 'interrupt' in sys.argv:
        asyncio.run(interrupt())
    if 'progress' in sys.argv:
        asyncio.run(progress())
    if 'options' in sys.argv:
        opt = options()
        log.debug({ 'options' })
        print(json.dumps(opt['options'], indent = 2))
        log.debug({ 'cmd-flags' })
        print(json.dumps(opt['flags'], indent = 2))
    if 'shutdown' in sys.argv:
        shutdown()
    asyncio.run(close())
