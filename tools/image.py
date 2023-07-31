#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 4:57 PM
# @Author  : wangdongming
# @Site    : 
# @File    : image.py
# @Software: Hifive
import os
import base64
import shutil
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo

Image.MAX_IMAGE_PIXELS = 933120000


def encode_pil_to_base64(image, quality=50):
    with BytesIO() as output_bytes:
        use_metadata = False
        metadata = PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None),
                   quality=quality)
        bytes_data = output_bytes.getvalue()

        return 'data:image/png;base64,' + base64.b64encode(bytes_data).decode('ascii')


# compress_image 压缩图片函数，减轻网络压力
def compress_image(infile, outfile, kb=400, step=25, quality=70):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 输出路径。
    :param kb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件字节流
    """
    o_size = os.path.getsize(infile) / 1024
    # print(f'  > 原始大小：{o_size}')
    if o_size <= kb:
        # 大小满足要求
        shutil.copy(infile, outfile)

    pnginfo_data = PngInfo()
    im = Image.open(infile)
    if hasattr(im, "text"):
        for k, v in im.text.items():
            pnginfo_data.add_text(k, str(v))

    im = im.convert("RGB")  # 兼容处理png和jpg
    img_bytes = None

    while o_size > kb:
        out = BytesIO()
        im.save(out, format="JPEG", quality=quality, pnginfo=pnginfo_data)
        if quality - step < 0:
            break
        img_bytes = out.getvalue()
        o_size = len(img_bytes) / 1024
        out.close()  # 销毁对象
        quality -= step  # 质量递减
    if img_bytes:
        with open(outfile, "wb+") as f:
            f.write(img_bytes)
    else:
        shutil.copy(infile, outfile)


def thumbnail(infile, outfile, scale=0.4, w=0, h=0, quality=70):
    img = Image.open(infile)
    if w == 0 or h == 0:
        w, h = img.size
        w, h = round(w * scale), round(h * scale)

    img.thumbnail((w, h))
    img.save(outfile, optimize=True, quality=quality)
    img.close()


def plt_show(img, title=None):
    import matplotlib.pyplot as plt
    plt.title(title or 'undefined')
    plt.imshow(img)
    plt.show()


