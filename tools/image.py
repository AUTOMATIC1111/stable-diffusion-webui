#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 4:57 PM
# @Author  : wangdongming
# @Site    : 
# @File    : image.py
# @Software: Hifive
import os
from io import BytesIO
from PIL import Image


# compress_image 压缩图片函数，减轻网络压力
def compress_image(infile, outfile, mb=400, step=15, quality=80):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 输出路径。
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件字节流
    """
    o_size = os.path.getsize(infile) / 1024
    # print(f'  > 原始大小：{o_size}')
    if o_size <= mb:
        with open(infile, 'rb') as f:
            content = f.read()
        return content  # 大小满足要求，直接返回字节流

    im = Image.open(infile)
    im = im.convert("RGB")  # 兼容处理png和jpg

    _imgbytes = None

    while o_size > mb:
        out = BytesIO()
        im.save(out, format="JPEG", quality=quality)
        if quality - step < 0:
            break
        _imgbytes = out.getvalue()
        o_size = len(_imgbytes) / 1024
        out.close()  # 销毁对象
        quality -= step  # 质量递减
    if _imgbytes:
        with open(outfile, "wb+") as f:
            f.write(_imgbytes)


def thumbnail(infile, outfile, scale=0.2):
    img = Image.open(infile)
    w, h = img.size
    w, h = round(w * scale), round(h * scale)

    img = img.resize((w, h), Image.ANTIALIAS)
    img.save(outfile, optimize=True, quality=85)

