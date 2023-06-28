import base64
import json
import threading
import requests

from scripts.extLogging import logger
from extra.loadYamlFile import ExtraConfig
from scripts.mqEntry import MqSupporter, QueueException
from pulsar import Message

from modules.api import models
from modules.api.api import Api
from fastapi import FastAPI
from modules.call_queue import queue_lock


def taskHandler(msg: Message, app: FastAPI):
    data = json.loads(msg.data())
    config = ExtraConfig().get_config()

    if msg.topic_name() == config["queue"]["topic-t2i"]:
        txt2imgreq = models.StableDiffusionTxt2ImgProcessingAPI(**data)
        logger.info("Text2Image Request '%s'", txt2imgreq)
        api = Api(app, queue_lock)
        response = api.text2imgapi(txt2imgreq)
        logger.info("Text2Image Result '%s'", response.dict())
        json_data = json.dumps(response.dict()).encode('utf-8')
        mq = MqSupporter()
        mq.createProducer(config["queue"]["topic-t2i-result"], json_data, msg.properties())
        mq.createProducer(f"{config['queue']['topic-web-img-result']}-{msg.properties()['userId']}", json_data, msg.properties())
    else:
        req = models.StableDiffusionImg2ImgProcessingAPI(**data)
        logger.info("Image2Image Request '%s'", req)
        resp = requests.get(f"{config['upload']['server-url']}/{req.init_images[0]}", stream=True)
        if resp.status_code == 200:
            encoded_file = base64.b64encode(resp.content).decode('utf-8')
            req.init_images = [encoded_file]
            api = Api(app, queue_lock)
            response = api.img2imgapi(req)
            logger.info("Image2Image Result '%s'", response.dict())
            json_data = json.dumps(response.dict()).encode('utf-8')
            mq = MqSupporter()
            mq.createProducer(config["queue"]["topic-i2i-result"], json_data, msg.properties())
            mq.createProducer(f"{config['queue']['topic-web-img-result']}-{msg.properties()['userId']}", json_data, msg.properties())
        else:
            logger.error("Image file download fail %s", f"{config['upload']['server-url']}/{req.init_images[0]}")


class TaskListener(threading.Thread):
    def __init__(self, app: FastAPI):
        super().__init__()
        self.app = app

    def run(self):
        config = ExtraConfig().get_config()
        mq = MqSupporter()
        mq.createConsumer(config["queue"]["topics"], config["queue"]["subscription"], config["queue"]["consumer-name"],
                          taskHandler, self.app)
        mq.closeClient()
