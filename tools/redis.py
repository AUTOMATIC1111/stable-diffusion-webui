#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : redis.py
# @Software: Hifive
import os
import redis
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

