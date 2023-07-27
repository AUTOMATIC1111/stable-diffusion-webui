#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 11:47 AM
# @Author  : wangdongming
# @Site    : 
# @File    : ticket.py
# @Software: Hifive
import os

Env_Ticket = "Ticket"


def get_ticket():
    return os.getenv(Env_Ticket, -1)

