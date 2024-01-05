#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 6:44 PM
# @Author  : wangdongming
# @Site    : 
# @File    : resp.py
# @Software: Hifive
import typing

from enum import IntEnum
from functools import wraps
from loguru import logger
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections.abc import Iterable


class StatusCode(IntEnum):

    OK = 200
    InternalError = 10000


class BaseResponse(JSONResponse):

    def __init__(
        self,
        data: typing.Any,
        code: StatusCode = StatusCode.OK,
        msg: str = 'ok',
        status_code: int = 200,
        headers: typing.Optional[typing.Dict[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional = None,
    ) -> None:
        content = {
            'code': code,
            'msg': msg,
            'data': data
        }
        super().__init__(content, status_code, headers, media_type, background)


def FastHandlerErrorDecorator(f: typing.Callable):
    @wraps(f)
    def wrapper(*args, **kwargs) -> BaseResponse:
        try:
            r = f(*args, **kwargs)
            if not isinstance(r, BaseResponse):
                if isinstance(r, BaseModel):
                    r = r.model_dump()
                elif isinstance(r, Iterable):
                    nr = []
                    for item in r:
                        if isinstance(item, BaseModel):
                            nr.append(item.model_dump())
                        else:
                            nr.append(item)
                    r = nr

                return BaseResponse(r)
            return r
        except Exception as ex:
            logger.exception(f'unhandle err :{ex}')
            return BaseResponse(str(ex), StatusCode.InternalError, 'internal error')
    return wrapper
