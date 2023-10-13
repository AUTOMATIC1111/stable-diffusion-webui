#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 9:54 AM
# @Author  : wangdongming
# @Site    : 
# @File    : handler.py
# @Software: Hifive
import abc
import time
import typing
import traceback
import torch.cuda
from importlib.metadata import version
from modules.shared import mem_mon as vram_mon
from worker.dumper import dumper
from loguru import logger
from modules.devices import torch_gc, get_cuda_device_string
from worker.k8s_health import write_healthy, system_exit
from worker.task import Task, TaskProgress, TaskStatus, TaskType
from modules import launch_utils
from enum import StrEnum


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


RequirementsType = typing.List[RequirementItem]


class TaskHandler:

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.enable = True
        self.requirements: RequirementsType = []

    def handle_task_type(self):
        return self.task_type

    @abc.abstractmethod
    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        raise NotImplementedError

    def _set_task_status(self, p: TaskProgress):
        logger.info(f">>> task:{p.task.desc()}, status:{p.status.name}, desc:{p.task_desc}")

    def do(self, task: Task, progress_callback=None):
        ok, msg = task.valid()
        if not ok:
            p = TaskProgress.new_failed(task, msg)
            self._set_task_status(p)
        else:
            try:
                p = TaskProgress.new_prepare(task, msg)
                self._set_task_status(p)
                for progress in self._exec(task):
                    self._set_task_status(progress)
                    if callable(progress_callback):
                        progress_callback(progress)

            except torch.cuda.OutOfMemoryError:
                torch_gc()
                time.sleep(15)
                logger.exception('CUDA out of memory')
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
                p = TaskProgress.new_failed(
                    task,
                    'CUDA out of memory',
                    f'CUDA out of memory and release, free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                self._set_task_status(p)
                progress_callback(p)
                system_exit(free, total)
            except Exception as ex:
                trace = traceback.format_exc()
                msg = str(ex)
                logger.exception('unhandle err')
                p = TaskProgress.new_failed(task, msg, trace)

                self._set_task_status(p)
                progress_callback(p)
                torch_gc()
                if 'BrokenPipeError' in str(ex):
                    pass

    def close(self):
        pass

    def get_installed_version(self, package: str) -> typing.Optional[str]:
        try:
            return version(package)
        except Exception:
            return None

    def comparable_version(self, version: str) -> typing.Tuple:
        return tuple(version.split('.'))

    def install_requirements(self):

        def tiny_pip_package(require: RequirementItem, compare_version_for_install: typing.Callable = None):
            package_name = require.package_name
            package_version = require.version

            installed_version = self.get_installed_version(package_name)
            if not installed_version or (
                    compare_version_for_install and compare_version_for_install(installed_version, package_version)):
                launch_utils.run_pip(f"install -U {require.require_expr}",
                                     f"{self.task_type.name} handler requirement: changing {package_name}"
                                     f" version from {installed_version} to {package_version}")

        if self.requirements:
            for require in self.requirements:
                package_name = require.package_name
                if require.require_expr == RequirementVersionExpr.Equal:
                    tiny_pip_package(require,
                                     lambda installed_version, package_version: installed_version != package_version)

                elif require.require_expr == RequirementVersionExpr.GreaterThanEqual:
                    tiny_pip_package(require,
                                     lambda installed_version, package_version: installed_version < package_version)
                elif require.require_expr == RequirementVersionExpr.GreaterThan:
                    tiny_pip_package(require,
                                     lambda installed_version, package_version: installed_version <= package_version)

                elif require.require_expr == RequirementVersionExpr.LessThan:
                    tiny_pip_package(require,
                                     lambda installed_version, package_version: installed_version >= package_version)

                elif require.require_expr == RequirementVersionExpr.LessThanEqual:
                    tiny_pip_package(require,
                                     lambda installed_version, package_version: installed_version > package_version)
                elif not launch_utils.is_installed(package_name):
                    launch_utils.run_pip(f"install {package_name}", f"sd-webui-controlnet requirement: {package_name}")

    def set_failed(self, task: Task, desc: str):
        p = TaskProgress.new_failed(task, desc)
        self._set_task_status(p)

    def __call__(self, *args, **kwargs):
        return self.do(*args, **kwargs)


class DumpTaskHandler(TaskHandler, abc.ABC):

    def __init__(self, task_type: TaskType):
        super(DumpTaskHandler, self).__init__(task_type)

    def _set_task_status(self, p: TaskProgress):
        super()._set_task_status(p)
        dumper.dump_task_progress(p)

    def close(self):
        dumper.stop()
