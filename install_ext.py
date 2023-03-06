#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 2:53 PM
# @Author  : wangdongming
# @Site    : 
# @File    : install_ext.py
# @Software: Hifive
import os
from modules.ui_extensions import install_extension_from_url

def setup():
    urls = [
        'https://github.com/Mikubill/sd-webui-controlnet',
        'https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN'
    ]

    for url in urls:
        basename = os.path.basename(url)
        print(f"install extentsion:{basename}")
        install_extension_from_url(None, url)

def safety_setup():
    try:
        setup()
    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    setup()