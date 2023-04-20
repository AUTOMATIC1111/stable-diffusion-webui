#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 3:24 PM
# @Author  : wangdongming
# @Site    : 
# @File    : multi-diffusion.py
# @Software: Hifive
import typing
from handlers.formatter import AlwaysonScriptArgsFormatter

Multidiffusion = "Tiled Diffusion"


class MultiDiffusionFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return Multidiffusion

    def format(self, is_img2img: bool, alwayson_scripts: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        pass
