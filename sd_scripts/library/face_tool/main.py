#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 6:16 PM
# @Author  : wangdongming
# @Site    : 
# @File    : app.py
# @Software: Hifive
import uvicorn
from fastapi import FastAPI
from data_models.types import BaseRoutePath
from app.routers.face import router as face_router
from fastapi.middleware.cors import CORSMiddleware
from tools.timer import Timer
from filestorage import clean_tmp_dir


origins = [
    "*"
]

fast_app = FastAPI()
fast_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
fast_app.include_router(face_router, prefix=BaseRoutePath)


@fast_app.get('/')
@fast_app.get('/index')
def index():
    return 'hello '


if __name__ == '__main__':
    timer = Timer()
    timer.add_job(clean_tmp_dir, 300)
    uvicorn.run('main:fast_app', host='0.0.0.0', port=12305)
    timer.stop()







