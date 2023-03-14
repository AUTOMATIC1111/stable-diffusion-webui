#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/10 11:23 AM
# @Author  : wangdongming
# @Site    : 
# @File    : user.py
# @Software: Hifive
import json
import os
import time
import typing

from tools.mysql import MySQLClient


def authorization(user, password):
    auths = {
        "wangdongming": "Admin123",
        "admin": "Admin123"
    }

    if user in auths:
        return auths.get(user) == password

    return find_users_from_models(user, password)


def find_users_from_models(username, password) -> bool:
    host = os.getenv('MysqlHost', '')
    user = os.getenv('MysqlUser', '')
    pwd = os.getenv('MysqlPass', '')
    db = os.getenv('MysqlDB', 'draw-ai')
    port = os.getenv('MysqlPort', 3306)

    if host:
        with MySQLClient(host, db, user, pwd, port) as cli:
            res = cli.query("SELECT * FROM user WHERE username=%s AND password=%s", (username, password))
            if res:
                expire = res.get('expire', -1)
                if 0 == expire or expire > time.time():
                    endpoint = os.getenv("Endpoint")
                    if endpoint and res.get('endpoint'):
                        return endpoint == res['endpoint']
                    return True
    return False

