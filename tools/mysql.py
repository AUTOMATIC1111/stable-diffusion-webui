#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 12:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : mysql.py
# @Software: Hifive
import os
import threading
import typing
import time
import pymysql
from tools.environment import get_mysql_env, Env_MysqlHost, \
    Env_MysqlPort, Env_MysqlPass, Env_MysqlUser, Env_MysqlDB


class MySQLClient(object):

    def __init__(self, addr=None, db=None, user=None, pwd=None, port=3306):
        settings = self.get_mysql_config(addr, port, db, user, pwd)
        self.conn = pymysql.connect(**settings)
        self._closed = False
        self._connection_time = time.time()
        self.config = settings

    @property
    def connect(self):
        return self.conn

    @property
    def available(self):
        return not self._closed

    @property
    def must_reconnct(self):
        return time.time() - self._connection_time > 10

    def check_connection(self):
        if self.available:
            if self.must_reconnct:
                self.conn.close()
                self.conn = pymysql.connect(**self.config)
                self._connection_time = time.time()

    def get_mysql_config(self, addr=None, port=3306, db=None, user=None, passwd=None):
        return {
            "host": "{}".format(addr),
            "user": "{}".format(user),
            "passwd": "{}".format(passwd),
            "db": "{}".format(db),
            "charset": "utf8",
            'port': int(port),
            'autocommit': True
        }

    def execute_noquery_cmd(self, cmd, args=None, connect=None, callback=None):
        connect = connect or self.connect
        with connect.cursor() as cursor:
            r = cursor.execute(cmd, args)
            if not cursor.description:
                return r
            if not connect.autocommit_mode:
                connect.commit()
            if callback:
                callback(cmd, args, connect, cursor)
            return r

    def execute_noquery_many(self, cmd, *args):
        connect = self.connect
        with connect.cursor() as cursor:
            r = cursor.executemany(cmd, list(args))
            if hasattr(connect, 'autocommit_mode') and not connect.autocommit_mode:
                connect.commit()
            return r

    def query(self, cmd, args=None, connect=None, fetchall=False):
        result = []

        def query_callback(cmd, args, connect, cursor):
            fields = [field_info[0] for field_info in cursor.description]
            if not fetchall:
                row = cursor.fetchone()
                if row:
                    res = {item[0]: item[1] for item in zip(fields, row)}
                    result.append(res)
                else:
                    return None
            else:
                res_list = [{item[0]: item[1] for item in zip(fields, info)} for info in cursor.fetchall()]
                result.extend(res_list)

        self.execute_noquery_cmd(cmd, args, connect, query_callback)
        if not result:
            return None
        elif not fetchall:
            return result[0]
        else:
            return result

    def transaction(self, *cmdArgs):
        connect = self.connect
        cursor = connect.cursor()
        try:
            for cmd, args in cmdArgs:
                cursor.execute(cmd, args)
        except Exception as ex:
            connect.rollback()
        else:
            if hasattr(connect, 'autocommit_mode') and not connect.autocommit_mode:
                connect.commit()
        finally:
            cursor.close()

    def close(self, conn=None):
        conn = conn or self.connect
        conn.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


mysql_clis = {}
locker = threading.RLock()


def get_mysql_cli(host: str = None, port: typing.Optional[int] = None, user: str = None,
                  pwd: str = None, db: str = None) -> typing.Optional[MySQLClient]:
    if os.getenv("ENABLE_TSS", "0") == '1':
        return None

    env_vars = get_mysql_env()
    if not host:
        host = env_vars.get(Env_MysqlHost)
    if not port:
        port = env_vars.get(Env_MysqlPort) or 3306
    if not user:
        user = env_vars.get(Env_MysqlUser) or 'root'
    if not pwd:
        pwd = env_vars.get(Env_MysqlPass)
    if not db:
        db = env_vars.get(Env_MysqlDB) or 'draw-ai'

    if not host:
        return None

    return MySQLClient(host, db, user, pwd, int(port))


def dispose():
    for cli in mysql_clis.values():
        if hasattr(cli, 'close'):
            cli.close()
