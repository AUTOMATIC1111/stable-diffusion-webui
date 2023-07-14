#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/14 6:03 PM
# @Author  : wangdongming
# @Site    : 
# @File    : test_fs.py
# @Software: Hifive
import os
import sys
import time
import unittest
from filestorage.oss import OssFileStorage


class TestFs(unittest.TestCase):

    def progress(self, a, b):
        print(a, b)

    def test_download(self):
        oss = OssFileStorage()
        oss.download(
            'xingzhe-sdplus/sd-web/output/53f082ae-28f6-40d1-8a45-a4b98fcffd3f/txt2img/grids/2023/07/14/low-t2i-o906w6elmgp5vy_60.png',
            'tmp/1.png',
            self.progress
        )


if __name__ == '__main__':

    unittest.main()