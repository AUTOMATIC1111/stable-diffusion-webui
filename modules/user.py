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
import base64
import os
import traceback

import requests
from hashlib import md5
from modules.shared import opts
from Crypto.Cipher import AES
from Crypto import Random
from tools.mysql import get_mysql_cli


def authorization(user, password):
    auths = {
        "admin": "Admin123",
    }

    if os.getenv("ENABLE_TSS", "0") == '1':
        print('tushuashua plugin enable!!!!')
        return request_tu(user, password)
    if user in auths and auths.get(user) == password:
        return 3600 * 24 + int(time.time())
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


def pad(data):
    length = 16 - (len(data) % 16)
    return (data + (chr(length) * length)).encode()


def unpad(data):
    return data[:-(data[-1] if type(data[-1]) == int else ord(data[-1]))]


def bytes_to_key(data: bytes, salt: bytes, output: int = 48):
    assert len(salt) == 8, len(salt)
    data += salt
    key = md5(data).digest()
    final_key = key
    while len(final_key) < output:
        key = md5(key + data).digest()
        final_key += key
    return final_key[:output]


def encrypt(message: str, passphrase: str):
    salt = Random.new().read(8)
    key_iv = bytes_to_key(passphrase.encode(), salt, 32+16)
    key = key_iv[:32]
    iv = key_iv[32:]
    aes = AES.new(key, AES.MODE_CBC, iv)
    return base64.b64encode(b"Salted__" + salt + aes.encrypt(pad(message)))


def request_tu(username, password):
    host = os.getenv('TSS_HOST', 'https://draw-plus-backend-qa.xingzheai.cn/')
    path = "/v1/login"
    bucket_path = "/v1/users/user_info"

    try:
        encrypted = encrypt(password, 'ZSYL20200707ZSYL')
        resp = requests.post(host + path, json={
            'account': username,
            'password': encrypted.decode()
        }, timeout=10)
        if resp:
            data = resp.json()
            if data.get('code', -1) == 200:
                token = data['data']['token']
                expire = time.time() + 10 * 3600
                setattr(opts, 'tu-token', {
                    'token': token,
                    'expire': expire,
                })

                bucket_resp = requests.get(host + bucket_path, headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {token}' if not token.startswith('Bearer') else token
                }, timeout=10)
                if bucket_resp:
                    data = bucket_resp.json()
                    if data.get('code', -1) == 200:
                        xz_bucket = data['data']['bucket']
                        setattr(opts, 'xz_bucket', xz_bucket)

                return expire

    except Exception as ex:
        traceback.print_exc()
        return -1

    return -1
