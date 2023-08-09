#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 4:01 PM
# @Author  : wangdongming
# @Site    : 
# @File    : storage.py
# @Software: Hifive
import abc
import os
import json
import shutil
import typing

import s3fs
import requests
import random
import importlib.util
from loguru import logger
from tools.processor import MultiThreadWorker
from multiprocessing import cpu_count
from urllib.parse import urlparse, urlsplit

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15',
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36"
]


def http_request(url, method='GET', headers=None, cookies=None, data=None, timeout=30, proxy=None, stream=False):
    _headers = {
        'Accept-Language': 'en-US, en; q=0.8, zh-Hans-CN; q=0.5, zh-Hans; q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html, application/xhtml+xml, image/jxr, */*',
        'Connection': 'Keep-Alive',
    }
    if headers and isinstance(headers, dict):
        _headers.update(headers)
    method = method.upper()
    kwargs = {
        'headers': _headers,
        'cookies': cookies,
        'timeout': timeout,
        'verify': False,
    }
    print(f"[{method}]request url:{url}")
    if method == 'GET':
        kwargs['stream'] = stream
    if proxy:
        kwargs['proxies'] = proxy
    if data and method != "GET" and 'json' in _headers.get('Content-Type', ''):
        data = json.dumps(data)
    scheme, netloc, path, query, fragment = urlsplit(url)
    if query:
        data = data or {}
        data.update((item.split('=', maxsplit=1) for item in query.split("&") if item))
        url = url.split("?")[0]

    res = None
    for i in range(3):
        try:
            if 'User-Agent' not in _headers:
                _headers['User-Agent'] = random.choice(USER_AGENTS)
            kwargs['headers'] = _headers
            if method == 'GET':
                res = requests.get(url, data, **kwargs)
            elif method == 'PUT':
                res = requests.put(url, data, **kwargs)
            elif method == 'DELETE':
                res = requests.delete(url, **kwargs)
            elif method == 'OPTIONS':
                data = data if isinstance(data, dict) else {}
                res = requests.options(url, **data)
            else:
                res = requests.post(url, data, **kwargs)
            if res.ok:
                break
        except:
            if i >= 2:
                raise
    return res


class FileStorage:

    def __init__(self):
        self.tmp_dir = os.path.join('tmp')
        os.makedirs(self.tmp_dir, exist_ok=True)

    @property
    def logger(self):
        return logger

    @abc.abstractmethod
    def download(self, remoting_path, local_path, progress_callback=None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def upload(self, local_path, remoting_path) -> str:
        raise NotImplementedError

    def upload_content(self, remoting_path, content) -> str:
        raise NotImplementedError

    def preview_url(self, remoting_path: str) -> str:
        raise NotImplementedError

    def multi_upload(self, local_remoting_pars: typing.Sequence[typing.Tuple[str, str]]):
        if local_remoting_pars:
            worker_count = cpu_count()
            worker_count = worker_count if worker_count <= 4 else 4
            w = MultiThreadWorker(local_remoting_pars, self.upload, worker_count)
            w.run()

    def multi_download(self, remoting_loc_pairs: typing.Sequence[typing.Tuple[str, str]]):
        if remoting_loc_pairs:
            worker_count = cpu_count()
            worker_count = worker_count if worker_count <= 4 else 4
            w = MultiThreadWorker(remoting_loc_pairs, self.download, worker_count)
            w.run()

    def close(self):
        pass

    def top_dir(self, p: str) -> str:
        array = p.strip(os.path.sep).split(os.path.sep)
        return array[0]

    def extract_buack_key_from_path(self, p: str) -> (str, str):
        array = p.strip('/').split('/')
        return array[0], '/'.join(array[1:])

    def mmie(self, p: str) -> str:
        _, ex = os.path.splitext(p)
        ex = ex.lower()
        mmie_d = {
            '.png': 'image/png',
            '.tar': 'application/x-tar',
            '.txt': 'text/plain',
            '.zip': 'application/zip',
            '.json': 'application/json',
            '.ico': 'image/vnd.microsoft.icon',
            '.jpeg': 'image/jpeg',
            '.jpg': 'image/jpeg',
        }
        if ex in mmie_d:
            return mmie_d[ex]
        return mmie_d['.txt']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PrivatizationFileStorage(FileStorage):

    def name(self):
        return 'default'

    def download(self, remoting_path: str, local_path: str, progress_callback=None) -> str:
        if os.path.isfile(local_path):
            return local_path

        # 如果是本地路径
        if os.path.isfile(remoting_path):
            shutil.copy(remoting_path, local_path)
            return local_path

        # http
        if 'http' not in remoting_path.lower():
            raise OSError(f'unsupported file:{remoting_path}')

        self.logger.info(f"download url: {remoting_path}...")
        resp = http_request(remoting_path)
        if resp:
            filename = os.path.basename(remoting_path)
            if 'Content-Disposition' in resp.headers:
                cd = resp.headers.get('Content-Disposition')
                map = dict((item.strip().split('=')[:2] for item in (item for item in cd.split(';') if '=' in item)))
                if 'filename' in map:
                    filename = map['filename'].strip('"')

            chunk_size = 512
            filepath = os.path.join(self.tmp_dir, filename)
            self.logger.info(f"save to {filename} ...")
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            if os.path.isdir(local_path):
                local_path = os.path.join(local_path, filename)
            shutil.move(filename, local_path)
            return local_path

    def upload(self, local_path, remoting_path) -> str:
        # local file system
        if remoting_path != local_path:
            shutil.copy(local_path, remoting_path)
            return remoting_path

    def upload_content(self, remoting_path, content) -> str:
        with open(remoting_path, 'wb+') as f:
            f.write(content)
        return remoting_path

    def preview_url(self, remoting_path: str) -> str:
        return remoting_path


