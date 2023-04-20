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
                return [args['enabled'],
                        args['multiply'],
                        args['weight'],
                        args['understand'],
                        args['layers'],
                        args['apply_to'],
                        args['start_steps'],
                        args['max_steps'],
                        args['up'],
                        args['up_aa'],
                        args['down'],
                        args['down_aa'],
                        args['intp'],
                        args['x'],
                        args['y'],
                        args['force_float']]
            return obj

        posex_script_args = args
        if is_img2img:
            if isinstance(args, dict):
                posex_script_args = obj_to_array(args)
            else:
                posex_script_args = [obj_to_array(x) for x in args]

        return posex_script_args
