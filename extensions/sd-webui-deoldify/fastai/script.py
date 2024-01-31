import os, sys, subprocess, inspect
from dataclasses import dataclass
from typing import Any
from argparse import ArgumentParser


@dataclass
class Param():
    "A parameter in a function used in `anno_parser` or `call_parse`"
    help:str=None
    type:type=None
    opt:bool=True
    action:str=None
    nargs:str=None
    const:str=None
    choices:str=None
    required:bool=None

    @property
    def pre(self): return '--' if self.opt else ''
    @property
    def kwargs(self): return {k:v for k,v in self.__dict__.items()
                              if v is not None and k!='opt'}

def anno_parser(func):
    "Look at params (annotated with `Param`) in func and return an `ArgumentParser`"
    p = ArgumentParser(description=func.__doc__)
    for k,v in inspect.signature(func).parameters.items():
        param = func.__annotations__.get(k, Param())
        kwargs = param.kwargs
        if v.default != inspect.Parameter.empty: kwargs['default'] = v.default
        p.add_argument(f"{param.pre}{k}", **kwargs)
    return p

def call_parse(func):
    "Decorator to create a simple CLI from `func` using `anno_parser`"
    name = inspect.currentframe().f_back.f_globals['__name__']
    if name == "__main__":
        args = anno_parser(func).parse_args()
        func(**args.__dict__)
    else: return func

def call_plac(f):
    "Decorator to create a simple CLI from `func` using `plac`"
    name = inspect.currentframe().f_back.f_globals['__name__']
    if name == '__main__':
        import plac
        res = plac.call(f)
        if callable(res): res()
    else: return f

