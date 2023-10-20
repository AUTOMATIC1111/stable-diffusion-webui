#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 6:59 PM
# @Author  : wangdongming
# @Site    : 
# @File    : install.py
# @Software: Hifive
import typing
from strenum import StrEnum
from worker.task import TaskType
from modules import launch_utils
from importlib.metadata import version


class RequirementVersionExpr(StrEnum):
    Equal = "=="
    GreaterThan = ">"
    LessThan = "<"
    GreaterThanEqual = ">="
    LessThanEqual = "<="


class RequirementItem:

    def __init__(self, name: str, ver: str = None, expr: RequirementVersionExpr = None):
        if ver and expr:
            self.require_expr = f"{name} {expr} {ver}"
        else:
            self.require_expr = f"{name}"
        self.package_name: str = name
        self.version: str = ver
        self.expr: RequirementVersionExpr = expr

    @classmethod
    def from_expr(cls, requirement_expr: str):
        def search_expr(s: str):
            for expr in ['==', '>=', '>', '<=', '<=']:
                if expr in s:
                    return expr

        expr = search_expr(requirement_expr)
        if expr:
            array = [x.strip() for x in requirement_expr.split(' ') if x.strip()]
        else:
            array = [requirement_expr.strip()]
        if len(array) < 1 or len(array) > 2:
            raise ValueError(f'requirement expr format err:{requirement_expr}')

        name = array[0]
        ver, expr = None, None
        if len(array) == 2:
            ver = RequirementVersionExpr(expr)
            expr = array[1]

        return cls(name, ver, expr)


Requirements: typing.Mapping[TaskType, typing.List[typing.Union[RequirementItem, str]]] = {
    TaskType.OnePress: [
        'onnxruntime-gpu',
        'insightface==0.7.1',

    ]
}

Repositories = {

}


def get_installed_version(package: str) -> typing.Optional[str]:
    try:
        return version(package)
    except Exception:
        return None


def comparable_version(ver: str) -> typing.Tuple:
    return tuple(ver.split('.'))


def install_pip_requirements():

    def tiny_pip_package(require: RequirementItem, task_type: TaskType,
                         compare_version_for_install: typing.Callable = None):
        package_name = require.package_name
        package_version = require.version

        installed_version = get_installed_version(package_name)
        if not installed_version or (
                compare_version_for_install and compare_version_for_install(installed_version, package_version)):
            launch_utils.run_pip(f"install -U {require.require_expr}",
                                 f"{task_type.name} handler requirement: changing {package_name}"
                                 f" version from {installed_version} to {package_version}")

    if Requirements:
        for task_type, requirements in Requirements.items():
            for require in requirements:
                require = require if isinstance(require, RequirementItem) else RequirementItem.from_expr(require)
                package_name = require.package_name
                if require.require_expr == RequirementVersionExpr.Equal:
                    tiny_pip_package(require, task_type,
                                     lambda installed_ver, package_ver: installed_ver != package_ver)

                elif require.require_expr == RequirementVersionExpr.GreaterThanEqual:
                    tiny_pip_package(require, task_type,
                                     lambda installed_ver, package_ver: installed_ver < package_ver)
                elif require.require_expr == RequirementVersionExpr.GreaterThan:
                    tiny_pip_package(require, task_type,
                                     lambda installed_ver, package_ver: installed_ver <= package_ver)

                elif require.require_expr == RequirementVersionExpr.LessThan:
                    tiny_pip_package(require, task_type,
                                     lambda installed_ver, package_ver: installed_ver >= package_ver)

                elif require.require_expr == RequirementVersionExpr.LessThanEqual:
                    tiny_pip_package(require, task_type,
                                     lambda installed_ver, package_ver: installed_ver > package_ver)
                elif not launch_utils.is_installed(package_name):
                    launch_utils.run_pip(f"install {package_name}",
                                         f"{task_type.name} handler requirement: {package_name}")



