#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 9:35 PM
# @Author  : wangdongming
# @Site    : 
# @File    : environment.py
# @Software: Hifive
import os
import typing
from tools.wrapper import FuncResultLogWrapper

S3ImageBucket = "xingzhe-sdplus"
S3ImagePath = "output/{uid}/{dir}/{name}"
S3Tmp = 'sd-tmp'
S3SDWEB = 'sd-web'

Env_MysqlHost = 'MysqlHost'
Env_MysqlPass = 'MysqlPass'
Env_MysqlPort = 'MysqlPort'
Env_MysqlUser = 'MysqlUser'
Env_MysqlDB = 'MysqlDB'
Env_RedisHost = 'RedisHost'
Env_RedisPass = 'RedisPass'
Env_RedisPort = 'RedisPort'
Env_RedisDB = 'RedisDB'
Env_MgoHost = 'MgoHost'
Env_MgoUser = 'MgoUser'
Env_MgoPass = 'MgoPass'
Env_MgoPort = 'MgoPort'
Env_MgoDB = 'MgoDB'
Env_MgoCollect = 'MgoCollect'
Env_EndponitKey = 'StorageEndponit'
Env_AccessKey = 'StorageAK'
Env_SecretKey = 'StorageSK'
Env_BucketKey = 'StorageBucket'

cache = {}


def get_mysql_env() -> typing.Mapping[str, str]:
    d = {
        Env_MysqlHost: None,
        Env_MysqlPort: None,
        Env_MysqlUser: None,
        Env_MysqlPass: None,
        Env_MysqlDB: None
    }
    for key in d.keys():
        d[key] = cache.get(key) or os.getenv(key)
        cache[key] = d[key]
    return d


def get_file_storage_system_env() -> typing.Mapping[str, str]:
    d = {
        Env_EndponitKey: None,
        Env_AccessKey: None,
        Env_SecretKey: None,
        Env_BucketKey: None,
    }
    for key in d.keys():
        d[key] = cache.get(key) or os.getenv(key)
        cache[key] = d[key]
    return d


def get_redis_env() -> typing.Mapping[str, str]:
    d = {
        Env_RedisPort: None,
        Env_RedisDB: None,
        Env_RedisPass: None,
        Env_RedisHost: None,
    }
    for key in d.keys():
        d[key] = cache.get(key) or os.getenv(key)
        cache[key] = d[key]
    return d


def get_mongo_env() -> typing.Mapping[str, str]:
    d = {
        Env_MgoHost: None,
        Env_MgoUser: None,
        Env_MgoPass: None,
        Env_MgoPort: None,
        Env_MgoDB: None,
        Env_MgoCollect: None,
    }
    for key in d.keys():
        d[key] = cache.get(key) or os.getenv(key)
        cache[key] = d[key]
    return d
