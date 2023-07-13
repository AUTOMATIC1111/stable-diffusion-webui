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
Env_RedisUser = 'RedisUser'
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

# 标记 WORKER执行TRAIN 的时机，不配做默认23~7点
Env_Run_Train_Time_Start = "RUN_TRAIN_TIME_START"
Env_Run_Train_Time_End = "RUN_TRAIN_TIME_END"
# 资源GROUP名称，如:aliyun
Env_Worker_Group = "WORKER_GROUP_NAME"
# 标记是否是弹性资源，有该标记就会认为是弹性资源
Env_Flexible_Res_Token = "FLEXIBLE_RESOURCE"
# 工作状态FILE记录路径
Env_Worker_State_File = "WORKER_STATE_FILE_PATH"
Env_GSS_Count_API = "GSS_COUNT_API"
Env_HostName = "hostname"
cache = {}


def is_flexible_worker():
    return os.getenv(Env_Flexible_Res_Token, "") != ""


def pod_host():
    return os.getenv(Env_HostName)


def get_gss_count_api():
    if is_flexible_worker():
        return os.getenv(Env_GSS_Count_API)


def get_worker_state_dump_path(defalut_path=None):
    return os.getenv(Env_Worker_State_File, defalut_path)


def get_worker_group():
    group = cache.get(Env_Worker_Group) or os.getenv(Env_Worker_Group, "Unknown")
    cache[Env_Worker_Group] = group

    return group


def get_run_train_time_cfg():
    d = {
        Env_Run_Train_Time_Start: None,
        Env_Run_Train_Time_End: None,
    }

    for key in d.keys():
        d[key] = cache.get(key) or os.getenv(key)
        cache[key] = d[key]
    return d


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
        Env_RedisUser: None
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
