#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 4:25 PM
# @Author  : wangdongming
# @Site    : 
# @File    : obs.py
# @Software: Hifive
import os
from obs import PutObjectHeader
from obs import ObsClient
from filestorage.storage import FileStorage
from tools.environment import get_file_storage_system_env, Env_EndponitKey, \
    Env_AccessKey, Env_SecretKey


class ObsFileStorage(FileStorage):

    def __init__(self):
        super(ObsFileStorage, self).__init__()
        env_vars = get_file_storage_system_env()
        endpoint = env_vars.get(Env_EndponitKey, '')
        access_key_id = env_vars.get(Env_AccessKey, '')
        secret_access_key = env_vars.get(Env_SecretKey, '')
        self.obsClient = None
        if 'huaweicloud' in endpoint:
            self.obsClient = ObsClient(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                server=endpoint
            )

    def name(self):
        return 'myhuaweicloud'

    def download(self, remoting_path, local_path) -> str:
        if self.obsClient and remoting_path and local_path:
            if os.path.isfile(local_path):
                return local_path
            bucket, key = self.extract_buack_key_from_path(remoting_path)
            self.logger.info(f"download {key} from obs to {local_path}")
            resp = self.obsClient.downloadFile(bucket, key, local_path, 10 * 1024 * 1024, 4, True)
            if resp.status < 300:
                return local_path
            else:
                raise OSError(f'cannot download file from obs, resp:{resp.errorMessage}, key: {remoting_path}')
        else:
            raise OSError('cannot init obs or file not found')

    def upload(self, local_path, remoting_path) -> str:
        if not os.path.isfile(local_path):
            raise OSError(f'cannot found file:{local_path}')
        bucket, key = self.extract_buack_key_from_path(remoting_path)
        headers = PutObjectHeader()
        headers.contentType = self.mmie(local_path)
        self.logger.info(f"upload file:{remoting_path}")
        # 分片上传
        resp = self.obsClient.uploadFile(bucket, key, local_path, 5*1024*1024, 4, True, headers=headers)

        if resp.status < 300:
            return remoting_path
        else:
            raise OSError(f'cannot download file from obs, resp:{resp.errorMessage},key: {remoting_path}')

