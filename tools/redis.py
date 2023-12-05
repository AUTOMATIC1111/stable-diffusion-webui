#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : redis.py
# @Software: Hifive
import os
import redis
import redis_lock

from tools.environment import get_redis_env, Env_RedisHost,\
    Env_RedisPort, Env_RedisDB, Env_RedisPass, Env_RedisUser


class RedisPool:
    def __init__(self, host=None, port=None, db=0, password=None, max_connections=10, user=None):
        env_vars = get_redis_env()
        if not host:
            host = env_vars.get(Env_RedisHost)
        if not port:
            port = env_vars.get(Env_RedisPort) or 6379
        if not password:
            password = env_vars.get(Env_RedisPass)
        if not db:
            db = env_vars.get(Env_RedisDB) or 1
        if not user:
            user = env_vars.get(Env_RedisUser)

        self.host = host
        self.port = int(port)
        self.db = int(db)
        self.password = password
        self.user = user
        self.max_connections = max_connections
        if not host:
            raise EnvironmentError('cannot found redis host')

        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            username=user,
            max_connections=self.max_connections)

    def get_connection(self):
        return redis.Redis(connection_pool=self.pool)

    def close(self):
        self.pool.disconnect()


class RedisLocker:

    def __init__(self, key, host=None, port=None, db=0, password=None, user=None, expire=None,
                 acq_blcok=True, acq_timeout=None):
        env_vars = get_redis_env()
        if not host:
            host = env_vars.get(Env_RedisHost)
        if not port:
            port = env_vars.get(Env_RedisPort) or 6379
        if not password:
            password = env_vars.get(Env_RedisPass)
        if not db:
            db = env_vars.get(Env_RedisDB) or 1
        if not user:
            user = env_vars.get(Env_RedisUser)

        self.host = host
        self.port = int(port)
        self.db = int(db)
        self.password = password
        self.user = user
        if not host:
            raise EnvironmentError('cannot found redis host')
        self.cli = redis.Redis(
            host, port, db, password, username=user,
        )
        self.key = f"locker:{key}"
        self.expire = expire
        self.locker = redis_lock.Lock(self.cli, self.key, expire=expire)
        self.acq_blcok = acq_blcok
        self.acq_timeout = acq_timeout

    def __enter__(self):
        self.locker.acquire(self.acq_blcok, self.acq_timeout)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.locker.release()
        self.cli.close()


def dist_locker(key, executor, expire=None, args=None, kwargs=None):
    with RedisLocker(key, expire=expire) as locker:
        args = args or []
        kwargs = kwargs or {}
        res = executor(*args, **kwargs)
        return res