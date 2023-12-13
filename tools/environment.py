#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 9:35 PM
# @Author  : wangdongming
# @Site    : 
# @File    : environment.py
# @Software: Hifive
import os
import typing
from tools.wrapper import timed_lru_cache

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
Env_MgoDocExp = "MgoDocExpSec"
Env_MgoCollect = 'MgoCollect'
Env_EndponitKey = 'StorageEndponit'
Env_AccessKey = 'StorageAK'
Env_SecretKey = 'StorageSK'
Env_BucketKey = 'StorageBucket'
Env_Ticket = "Ticket"

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
Env_TaskGroupQueueOnly = "WORKER_GROUP_QUEUE_ONLY"
Env_WorkerRunTrainRatio = "RUN_TRAIN_RATIO"
# 不开启定期清除未使用模型文件
Env_DontCleanModels = "DONT_CLEAN_MODELS"
# 谛听审核APP KEY
Env_DtAppKey = "DT_APPKEY"
# 下载启用文件锁
Env_DownloadLocker = "DOWNLOAD_LOCKER"
# 维护模式key
Env_Maintain = "MAINTAIN"

cache = {}


def enable_download_locker():
    v = cache.get(Env_DownloadLocker, os.getenv(Env_DownloadLocker, '1'))
    cache[Env_DownloadLocker] = v
    return v == "1"


def is_flexible_worker():
    return os.getenv(Env_Flexible_Res_Token, "") != ""


def pod_host():
    return os.getenv(Env_HostName)


def mongo_doc_expire_seconds():
    '''
    获取MONGO文档过期时间，0代表不过期
    '''
    try:
        exp = os.getenv(Env_MgoDocExp, 0)
        exp = int(exp)
    except:
        exp = 0
    return exp


def run_train_ratio():
    v = os.getenv(Env_WorkerRunTrainRatio, 0.8)
    v = float(v)
    return 0.8 if v > 0.8 else v


def get_gss_count_api():
    if is_flexible_worker():
        return os.getenv(Env_GSS_Count_API)


def get_worker_state_dump_path(defalut_path=None):
    return os.getenv(Env_Worker_State_File, defalut_path)


def get_worker_group():
    group = cache.get(Env_Worker_Group) or os.getenv(Env_Worker_Group, "Unknown")
    cache[Env_Worker_Group] = group

    return group


def is_task_group_queue_only():
    x = cache.get(Env_TaskGroupQueueOnly) or os.getenv(Env_TaskGroupQueueOnly)
    cache[Env_TaskGroupQueueOnly] = x

    return x == "1"


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


def get_maintain_env():
    return os.getenv(Env_Maintain, "default")


def get_ticket():
    return os.getenv(Env_Ticket, -1)
