#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : redis.py
# @Software: Hifive
import os
import redis


class RedisPool:
    def __init__(self, host=None, port=None, db=0, password=None, max_connections=10):

        if not host:
            host = os.getenv('RedisHost')
        if not port:
            port = os.getenv('RedisPort', 6379)
        if not password:
            password = os.getenv('RedisPass')
        if not db:
            db = os.getenv('RedisDB', 1)
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        if not host:
            raise EnvironmentError('cannot found redis host')

        self.pool = redis.ConnectionPool(
            host=self.host, port=self.port, db=self.db, password=self.password, max_connections=self.max_connections)

    def get_connection(self):
        return redis.Redis(connection_pool=self.pool)

    def close(self):
        self.pool.disconnect()

