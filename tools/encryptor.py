#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 4:05 PM
# @Author  : wangdongming
# @Site    : 
# @File    : encryptor.py
# @Software: Hifive
import base64
import os.path

from Crypto.Cipher import DES
from Crypto.Util.Padding import pad

DefaultDESKey = 'A3094221'
DefaultDESIV = '87651234'
DefaultDESMode = DES.MODE_CBC


def des_encrypt(text, mode=None, key=None, iv=None):
    '''
    DES加密
    :param key: 密钥
    :param text: 待加密文本
    :param mode: 模式
    :param iv: iv
    :return:
    '''
    key = key or DefaultDESKey
    key = key.encode('utf-8')
    text = text.encode('utf-8')
    iv = iv or DefaultDESIV
    mode = mode or DefaultDESMode
    pad_text = pad(text, DES.block_size, style='pkcs7')
    cipher = DES.new(key, mode, iv=iv.encode('utf-8'))
    enc_data = cipher.encrypt(pad_text)
    return base64.b64encode(enc_data).decode('utf-8')


def des_decrypt(text, mode=None, key=None, iv=None):
    '''
    DES解密
    :param key: 密钥
    :param text: 待解密文本
    :param mode: 模式
    :param iv: iv
    :return:
    '''
    key = key or DefaultDESKey
    mode = mode or DefaultDESMode
    new_key = key.encode('utf-8')
    iv = iv or DefaultDESIV

    new_text = base64.b64decode(text)  # base64解码
    cipher = DES.new(new_key, mode, iv=iv)
    dec_data = cipher.decrypt(new_text)

    return dec_data[:-dec_data[-1]].decode('utf-8')  # 去除末尾填充的字符


def string_to_hex(text: str) -> bytes:
    b = bytes.hex(text.encode('utf8'))
    return b.encode('utf8')


def b64_image(image_path: str):
    _, ext = os.path.splitext(image_path)
    with open(image_path, "rb") as f:
        img = f.read()

    data = base64.b64encode(img).decode()
    return f"data:image/{ext};base64,{data}"


if __name__ == "__main__":
    key = '12345678'
    pwd = '12223333'
    enc_pwd = des_encrypt(key, pwd, DES.MODE_ECB)
    print(enc_pwd)
    dec_data = des_decrypt(key, enc_pwd, DES.MODE_ECB)
    print(dec_data)
