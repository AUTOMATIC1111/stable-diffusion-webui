# celery_workers.py
# Created on: 2023-12-22 14:57:07
# Author: VuLe@macbook
# Last updated: 2023-12-22 10:27:53
# Last modified by: VuDLe@Tu Nong's server

from celery.app import Celery
import time
import subprocess
from utils import general
import os
import time
import asyncio
import yaml
import random
from workers import api

# logger = general.setup_logging()

CELERY_BROKER_URL = "redis://205.134.227.4:6379"
CELERY_RESULT_BACKEND = "redis://205.134.227.4:6379"


celery = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)


@celery.task(name="create_task")
def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return True


@celery.task
def add(x, y):
    print(f"adding {x} and {y} got {x+y}")
    return x + y


@celery.task
def mul(x, y):
    return x * y


@celery.task
def sleep(seconds: int):
    print(f"sleeping for {seconds} seconds")
    time.sleep(seconds)


@celery.task
def call_api():
    pass


@celery.task(name="text to image diffusion")
def txt2image(payload, webui_server_url="http://127.0.0.1:7860"):
    response = api.call_api(webui_server_url, "sdapi/v1/txt2img", **payload)
    return response
