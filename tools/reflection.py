#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/1 3:03 PM
# @Author  : wangdongming
# @Site    :
# @File    : reflection.py
# @Software: Hifive
import os
import pkgutil
import importlib.util
from importlib import import_module
from inspect import getmembers, isclass, ismethod


def dynamic_import(module_path, package=None):
    """
    动态导入模块，并返回模块的实例。
    :param package:
    :param module_path: 模块的全路径。
    :package: 包名，如果module_path没有指定模块所在的包时使用。
    :return: 实例。
    """
    return import_module(module_path, package)


def load_object(path):
    """Load an object given its absolute object path, and return it.

    object can be a class, function, variable or an instance.
    path ie: 'remmsys_bp.layers.reallayers.filterlayer.Weather'
    """

    try:
        dot = path.rindex('.')
    except ValueError:
        raise ValueError("Error loading object '%s': not a full path" % path)

    module, name = path[:dot], path[dot + 1:]
    mod = import_module(module)

    try:
        obj = getattr(mod, name)
    except AttributeError:
        raise NameError("Module '%s' doesn't define any object named '%s'" % (module, name))

    return obj


def dynamic_create_ins(module_path, class_name, package=None, kwargs=None):
    kwargs = kwargs or {}
    module = dynamic_import(module_path, package)
    for (_, c) in getmembers(module):
        if isclass(c) and class_name == c.__name__:
            return c(**kwargs)


def find_classes(module_path):
    '''
    动态查找模块下的类，可用于动态创建指定类型的实例。
    :param module_path: 模块路径。
    :return:
    '''
    path = module_path.replace('.', "/")
    for importer, name, ispkg in pkgutil.walk_packages([f'./{path}']):
        module = import_module(f'{module_path}.{name}')
        for (_, c) in getmembers(module):
            if isclass(c):
                yield c


def load_module(path):
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def find_methods(module_path):
    '''
     动态查找模块下的function。
    :param module_path: 模块路径。
    :return:
    '''
    path = module_path.replace('.', "/")
    for importer, name, ispkg in pkgutil.walk_packages([f'./{path}', ]):
        module = import_module(f'{module_path}.{name}')
        for (_, c) in getmembers(module):
            if ismethod(c):
                yield c

