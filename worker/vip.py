#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/6 6:45 PM
# @Author  : wangdongming
# @Site    : 
# @File    : vip.py
# @Software: Hifive
from enum import IntEnum


class VipLevel(IntEnum):

    Level_1 = 0    # normal
    Level_10 = 10  # vip
    Level_99 = 99  # svip
