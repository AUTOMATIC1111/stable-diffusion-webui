from operator import div as _div_
from inspect import getargspec

def raise_with_traceback(exc, tb):
    raise exc, None, tb

__all__ = ['_div_', 'getargspec', 'raise_with_traceback']
