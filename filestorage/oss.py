#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 4:25 PM
# @Author  : wangdongming
# @Site    : 
# @File    : oss.py
# @Software: Hifive
import os
import shutil
import uuid
import oss2
from tools.redis import RedisLocker
from filestorage.storage import FileStorage
from tools.processor import MultiThreadWorker
from tools.environment import get_file_storage_system_env, Env_EndponitKey, \
    Env_AccessKey, Env_SecretKey, Env_BucketKey


def download(obj, bucket, local_path, tmp):
    if not os.path.isfile(local_path):
        tmp_file = os.path.join(tmp, os.path.basename(obj.key))
        oss2.resumable_download(bucket, obj.key, tmp_file)
        if os.path.isfile(tmp_file) and not os.path.isfile(local_path):
            shutil.move(tmp_file, local_path)


class OssFileStorage(FileStorage):

    def __init__(self):
        super(OssFileStorage, self).__init__()
        env_vars = get_file_storage_system_env()
        endpoint = env_vars.get(Env_EndponitKey, '')
        access_key_id = env_vars.get(Env_AccessKey, '')
        secret_access_key = env_vars.get(Env_SecretKey, '')
        bucket = env_vars.get(Env_BucketKey, '')

        self.bucket_name = bucket
        self.client = None
        if 'aliyun' in endpoint:
            self.endpoint = endpoint
            self.auth = oss2.Auth(access_key_id, secret_access_key)
            self.bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        else:
            self.auth = None
            self.bucket = None

    def name(self):
        return 'aliyuncs'

    def download(self, remoting_path, local_path, progress_callback=None) -> str:
        if self.auth and remoting_path and local_path:
            if os.path.isfile(local_path):
                return local_path
            if os.path.isfile(remoting_path):
                return remoting_path

            key = self.get_keyname(remoting_path, self.bucket_name)
            # bucket, key = self.extract_buack_key_from_path(remoting_path)
            self.logger.info(f"download {key} from oss to {local_path}")
            tmp_file = os.path.join(self.tmp_dir, "tmp-" + os.path.basename(local_path))
            # bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
            # f = self.acquire_flock(local_path)
            # acquire file locker
            if os.path.isfile(local_path):
                return local_path

            oss2.resumable_download(self.bucket, key, tmp_file, progress_callback=progress_callback)

            if os.path.isfile(local_path):
                return local_path

            if os.path.isfile(tmp_file):
                shutil.move(tmp_file, local_path)
                return local_path
            else:

                raise OSError(f'cannot download file from oss, {remoting_path}')

        else:
            raise OSError('cannot init oss or file not found')

    def upload(self, local_path, remoting_path) -> str:
        if not os.path.isfile(local_path):
            raise OSError(f'cannot found file:{local_path}')
        # bucket, key = self.extract_buack_key_from_path(remoting_path)
        key = self.get_keyname(remoting_path, self.bucket_name)
        self.logger.info(f"upload file:{remoting_path}")
        # bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        # 分片上传
        headers = oss2.CaseInsensitiveDict()
        headers['Content-Type'] = self.mmie(local_path)
        resp = self.bucket.put_object_from_file(key, local_path)

        if resp.status < 300:
            return remoting_path
        else:
            raise OSError(f'cannot download file from oss, resp:{resp.errorMessage}, key: {remoting_path}')

    def upload_content(self, remoting_path, content) -> str:
        # bucket, key = self.extract_buack_key_from_path(remoting_path)
        key = self.get_keyname(remoting_path, self.bucket_name)
        self.logger.info(f"upload file:{remoting_path}")
        # bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        # 分片上传
        headers = oss2.CaseInsensitiveDict()
        resp = self.bucket.put_object(key, content, headers)
        if resp.status < 300:
            return remoting_path
        else:
            raise OSError(f'cannot download file from oss, resp:{resp.errorMessage}, key: {remoting_path}')

    def preview_url(self, remoting_path: str) -> str:
        # bucket, key = self.extract_buack_key_from_path(remoting_path)
        # bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        return self.bucket.sign_url('GET', remoting_path, 10 * 60)

    def download_dir(self, remoting_dir: str, local_dir: str) -> bool:
        super(OssFileStorage, self).download_dir(remoting_dir, local_dir)
        # bucket_name, key = self.extract_buack_key_from_path(remoting_dir)
        # bucket = oss2.Bucket(self.auth, self.endpoint, bucket_name)

        tmp = os.path.join(self.tmp_dir, str(uuid.uuid4()))
        os.makedirs(tmp, exist_ok=True)

        args = []
        # 获取OBJ列表
        for i, obj in enumerate(oss2.ObjectIteratorV2(self.bucket, prefix=remoting_dir)):
            if i >= 200:
                break
            local_path = os.path.join(local_dir, os.path.basename(obj.key))
            args.append((obj, self.bucket, local_path, tmp))

        # 下载列表~
        worker = MultiThreadWorker(args, download, 4)
        worker.run()

        for item in args:
            local_path = item[2]
            if not os.path.isfile(local_path):
                shutil.rmtree(tmp)

                return False

        return True

    def lock_download(self, remoting_path, local_path, progress_callback=None, expire=1800, flocker=True) -> str:
        if os.path.isfile(local_path):
            return local_path
        if os.path.isfile(remoting_path):
            return remoting_path
        locker_key = self.get_lock_key(remoting_path)
        key = self.get_keyname(remoting_path, self.bucket_name)
        tmp_file = os.path.join(self.tmp_dir, "tmp-" + os.path.basename(local_path))

        if os.path.isfile(local_path):
            return local_path
        if not flocker:
            with RedisLocker(locker_key, expire=expire):
                if os.path.isfile(local_path):
                    return local_path
                self.logger.info(f"download (with dist locker:{locker_key}) {key} from oss to {local_path}")
                oss2.resumable_download(self.bucket, key, tmp_file, progress_callback=progress_callback)

                if os.path.isfile(local_path):
                    return local_path

                if os.path.isfile(tmp_file):
                    shutil.move(tmp_file, local_path)
                    return local_path
                else:
                    raise OSError(f'cannot download file from oss, {remoting_path}')
        else:
            f = None
            try:
                f = self.acquire_flock(remoting_path, local_path, timeout=expire)
                if os.path.isfile(local_path):
                    return local_path
                self.logger.info(f"download (with file locker) {key} from oss to {local_path}")
                oss2.resumable_download(self.bucket, key, tmp_file, progress_callback=progress_callback)

                if os.path.isfile(local_path):
                    return local_path

                if os.path.isfile(tmp_file):
                    shutil.move(tmp_file, local_path)
                    return local_path
                else:
                    raise OSError(f'cannot download file from oss, {remoting_path}')
            except:
                raise
            finally:
                self.release_flock(f, remoting_path)
