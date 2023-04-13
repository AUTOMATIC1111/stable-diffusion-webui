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

    def format(self, alwayson_scripts: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        return alwayson_scripts


formatters = {}


def init_formatters():
    for cls in find_classes('handlers'):
        if issubclass(cls, AlwaysonScriptArgsFormatter):
            ins = cls()
            if not ins.name():
                continue

            formatters[ins.name()] = ins.format


def format_alwayson_script_args(name, args):
    if not formatters:
        init_formatters()
    formatter = formatters.get(name)
    if not formatter:
        return args
    return formatter(args)
