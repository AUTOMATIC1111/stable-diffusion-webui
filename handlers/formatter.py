#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/6 6:11 PM
# @Author  : wangdongming
# @Site    : 
# @File    : formatter.py
# @Software: Hifive
import typing
from tools.reflection import find_classes


class AlwaysonScriptArgsFormatter:

    def name(self):
        return 'base'

    def format(self, is_img2img: bool, alwayson_scripts: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        return alwayson_scripts


formatters = {

}


def init_formatters():
    for cls in find_classes('handlers.extension'):
        if issubclass(cls, AlwaysonScriptArgsFormatter):
            ins = cls()
            name = ins.name()
            print(f"load {name} formatter")
            if not ins.name() or name in formatters:
                continue
            print(f"add {name} formatter")
            formatters[ins.name()] = ins.format


def format_alwayson_script_args(name, is_img2img, args):
    if not formatters:
        init_formatters()

    format = formatters.get(name)
    if not format:
        return args
    return format(is_img2img, args)
