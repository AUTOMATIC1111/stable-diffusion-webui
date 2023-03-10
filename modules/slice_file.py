#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 5:48 PM
# @Author  : wangdongming
# @Site    : 
# @File    : slice_file.py
# @Software: Hifive
import io
import json
import os
import math
import shutil
import hashlib
import typing

import requests
from requests_toolbelt import MultipartEncoder

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 2


class SliceFileMeta(dict):

    def __init__(self, md5, total_chunk, total_size, filename, seq=None, **kwargs):
        super(dict, self).__init__(seq, **kwargs)
        self['total_size'] = total_size
        self['filename'] = filename
        self['total_chunk'] = total_chunk
        self['md5'] = md5

    @property
    def md5(self):
        return self.get('md5')

    @property
    def filename(self):
        return self.get('filename')

    @property
    def total_size(self):
        return self.get('total_size')

    @property
    def total_chunk(self):
        return self.get('total_chunk')

    def validation(self, file_path: str):
        if not os.path.isfile(file_path):
            raise Exception(f'file not found:{file_path}')
        file_size = os.path.getsize(file_path)
        if self.total_size != file_size:
            raise Exception(f'file size err: file size {file_size} bytes,expect {self.total_size} bytes')
        if not self.md5:
            raise Exception(f'file hash not found:{self.filename}')
        hash_str = get_md5(file_path)
        if hash_str != self.md5:
            raise Exception(f'file hash code err: file code {hash_str},expect {self.md5}')

    @classmethod
    def from_dict(cls, d: typing.Mapping):
        md5 = d.get('md5')
        total_chunk = d.get('total_chunk')
        total_size = d.get('total_size')
        filename = d.get('filename')
        return cls(md5, total_chunk, total_size, filename)


def upload_slice_file(url: str, file_path: str, chunk_size: int = 0):
    if chunk_size < 1:
        chunk_size = DEFAULT_CHUNK_SIZE
    filename = file_path.split("\\")[-1:][0]
    total_size = os.path.getsize(file_path)
    current_chunk = 1
    total_chunk = math.ceil(total_size / chunk_size)

    while current_chunk <= total_chunk:
        start = (current_chunk - 1) * chunk_size
        end = min(total_size, start + chunk_size)
        with open(file_path, 'rb') as f:
            f.seek(start)
            file_chunk_data = f.read(end - start)
        data = MultipartEncoder(
            fields={
                "filename": filename,
                "totalSize": str(total_size),
                "currentChunk": str(current_chunk),
                "totalChunk": str(total_chunk),
                "md5": get_md5(file_path),
                "file": (filename, file_chunk_data, 'application/octet-stream')
            }
        )
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Content-Type": data.content_type
        }

        with requests.post(url, headers=headers, data=data) as response:
            assert response.status_code == 200

        current_chunk = current_chunk + 1
        print(f"upload progress:{current_chunk * 100 // current_chunk}%")


def get_md5(path: str) -> str:
    m = hashlib.md5()
    with open(path, 'rb') as f:
        for line in f:
            m.update(line)
    md5code = m.hexdigest()
    return md5code


def hashs(s: str) -> str:
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


def receive_slice_file():
    pass


def get_fragment_name(filename: str, chunk: int):
    name = os.path.basename(filename)
    return f'{name}_{chunk}'


def get_slice_meta_json_path(folder: str, filename: str) -> str:
    tmp_dir = get_tmp_slice_file(folder, filename)
    return os.path.join(tmp_dir, "meta.json")


def get_tmp_slice_file(folder: str, filename: str) -> str:
    tmp_dir = hashs(filename)
    return os.path.join(folder, tmp_dir)


def load_slice_meta_json(folder: str, filename: str) -> typing.Optional[SliceFileMeta]:
    json_path = get_slice_meta_json_path(folder, filename)
    if os.path.isfile(json_path):
        d = json.loads(json_path)
        return d


def write_slice_meta_json(folder: str, filename: str, meta: SliceFileMeta):
    json_path = get_slice_meta_json_path(folder, filename)
    with open(json_path, "w+") as f:
        s = json.dumps(meta)
        f.write(s)


def merge_slice_files(folder: str, filename: str, file_type: str):
    name = os.path.basename(filename)
    target_file_name = os.path.join(folder, f'{name}.{file_type}')
    tmp = get_tmp_slice_file(folder, filename)

    try:
        meta = load_slice_meta_json(folder, filename)
        with open(target_file_name, 'wb+') as target_file:  # 打开目标文件
            for i in range(1, meta.total_chunk):
                temp_file_name = os.path.join(tmp, get_fragment_name(name, i))
                with open(temp_file_name, 'rb') as temp_file:  # 按序打开每个分片
                    data = temp_file.read()
                    target_file.write(data)  # 分片内容写入目标文件
        meta.validation(target_file_name)
    except Exception as e:
        return {
            'code': -1,
            'error': f'merge file failed：{e}'
        }

    shutil.rmtree(tmp)  # 删除临时目录


def write_fragment_file(meta: SliceFileMeta, stream: io.BytesIO, current_chunk: int):
    pass
