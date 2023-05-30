#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 1:59 PM
# @Author  : wangdongming
# @Site    :
# @File    : llul.py
# @Software: Hifive
import os.path
import typing
from handlers.formatter import AlwaysonScriptArgsFormatter

LLuL = 'LLuL'


class LLuLFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return LLuL

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        def obj_to_array(obj: typing.Mapping) -> typing.Sequence:
            # 如果是[OBJ1, OBJ2]形式的，需要转换为ARRAY
            if isinstance(obj, dict):
                return [obj['enabled'],
                        obj['multiply'],
                        obj['weight'],
                        obj['understand'],
                        obj['layers'],
                        obj['apply_to'],
                        obj['start_steps'],
                        obj['max_steps'],
                        obj['up'],
                        obj['up_aa'],
                        obj['down'],
                        obj['down_aa'],
                        obj['intp'],
                        str(obj['x']),
                        str(obj['y']),
                        obj['force_float']]
            return obj

        llul_script_args = args
        if isinstance(args, dict):
            llul_script_args = obj_to_array(args)
        else:
            llul_script_args = []
            for x in args:
                llul_script_args.extend(obj_to_array(x))

        return llul_script_args


