#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/10 11:23 AM
# @Author  : wangdongming
# @Site    : 
# @File    : user.py
# @Software: Hifive
import json
import os.path


def authorization(user, password):
    auths = {
        "wangdongming": "Admin123",
        "admin": "Admin123"
    }

    if user in auths:
        return auths.get(user) == password

    return find_users_from_models(user, password)


def find_users_from_models(user, password) -> bool:
    cfg = 'models/user.json'
    if os.path.isfile(cfg):
        with open(cfg) as f:
            lines = f.readlines()
            try:
                d = json.loads("".join(lines))
                if user in d:
                    return d[user] == password
            except:
                return False
    return False

