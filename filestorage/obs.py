#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 4:25 PM
# @Author  : wangdongming
# @Site    : 
# @File    : obs.py
# @Software: Hifive
import os
from obs import PutObjectHeader
from filestorage.storage import FileStorage, EndponitKey, AccessKey, SecretKey
from obs import ObsClient


class ObsFileStorage(FileStorage):

    def __init__(self):
        super(ObsFileStorage, self).__init__()
        endpoint = os.getenv(EndponitKey, '')
        access_key_id = os.getenv(AccessKey, '')
        secret_access_key = os.getenv(SecretKey, '')
        self.obsClient = None
        if 'myhuaweicloud' in endpoint:
            self.obsClient = ObsClient(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                server=endpoint
            )

    def name(self):
        return 'myhuaweicloud'

    def download(self, remoting_path, local_path) -> str:
        if self.obsClient and remoting_path and local_path:
            bucket, key = self.extract_buack_key_from_path(remoting_path)
            taskNum = 4
            partSize = 5 * 1024 * 1024
            enableCheckpoint = True
            resp = self.obsClient.downloadFile(bucket, key, local_path, partSize, taskNum, enableCheckpoint)
            if resp.status < 300:
                return local_path
            else:
                raise OSError(f'cannot download file from obs, resp:{resp.errorMessage}')
        else:
            raise OSError('cannot init obs or file not found')

    def upload(self, local_path, remoting_path) -> str:
        if not os.path.isfile(local_path):
            raise OSError(f'cannot found file:{local_path}')
        bucket, key = self.extract_buack_key_from_path(remoting_path)
        headers = PutObjectHeader()
        headers.contentType = self.mmie(local_path)

        resp = self.obsClient.putFile(bucket, key, local_path, headers=headers)

        if resp.status < 300:
            return remoting_path
        else:
            raise OSError(f'cannot download file from obs, resp:{resp.errorMessage}')

