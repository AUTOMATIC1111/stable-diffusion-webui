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

import requests
from modules.shared import opts
from tools.mysql import get_mysql_cli


def authorization(user, password):
    auths = {
        "admin": "Admin123",
    }

    if user in auths and auths.get(user) == password:
        return 3600 * 24 + int(time.time())

    if os.getenv("ENABLE_TSS", "0") == '1':
        return request_tu(user, password)

    return find_users_from_db(user, password)


def find_users_from_db(username, password) -> int:
    host = os.getenv('MysqlHost')
    if host:
        cli = get_mysql_cli()
        try:
            res = cli.query("SELECT * FROM user WHERE username=%s AND password=%s", (username, password))
            if res:
                expire = res.get('expire', -1)
                if 0 == expire or expire > time.time():
                    endpoint = os.getenv("Endpoint")
                    if endpoint and res.get('endpoint'):
                        if endpoint != res['endpoint']:
                            return -1
                    return expire
        except Exception as ex:
            print(ex)
            return -1
        finally:
            cli.close()
    return -1


def request_tu(username, password):
    host = 'https://draw-plus-backend-qa.xingzheai.cn'
    path = "/v1/login"

    try:
        resp = requests.post(host+path, json={
            'account': username,
            'password': password
        }, timeout=10)
        if resp:
            data = resp.json()
            if data.get('code', -1) == 200:
                token = data['token']
                expire = time.localtime() + 10 * 3600
                setattr(opts, 'tu-token', {
                    'token': token,
                    'expire': expire,
                })
                return expire

    except:
        return -1

    return -1

