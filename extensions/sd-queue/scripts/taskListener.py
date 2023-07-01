import base64
import json
import threading

from scripts.extLogging import logger
from scripts.mqEntry import MqSupporter

from extra.fileStorage import ExtraFileStorage
from extra.loadYamlFile import ExtraConfig
from modules.api import models
from modules.api.api import Api
from modules.call_queue import queue_lock
from pulsar import Message


def taskHandler(msg: Message):
    from fastapi import FastAPI
    data = json.loads(msg.data())
    config = ExtraConfig("prod").get_config()

    if msg.topic_name() == config["queue"]["topic-t2i"]:
        txt2imgreq = models.StableDiffusionTxt2ImgProcessingAPI(**data)
        logger.info("Text2Image Request '%s'", txt2imgreq)
        app = FastAPI()
        api = Api(app, queue_lock)
        response = api.text2imgapi(txt2imgreq)
        logger.info("Text2Image Result '%s'", response.dict())
        json_data = json.dumps(response.dict()).encode('utf-8')
        mq = MqSupporter()
        mq.createProducer(config["queue"]["topic-t2i-result"], json_data, msg.properties())
        mq.createProducer(f"{config['queue']['topic-web-img-result']}-{msg.properties()['userId']}", json_data,
                          msg.properties())
    else:
        req = models.StableDiffusionImg2ImgProcessingAPI(**data)
        logger.info("Image2Image Request '%s'", req)

        try:
            storage = ExtraFileStorage()
            resp = storage.downloadFile(req.init_images[0])
            encoded_file = base64.b64encode(resp.read()).decode('utf-8')
            req.init_images = [encoded_file]
            app = FastAPI()
            api = Api(app, queue_lock)
            response = api.img2imgapi(req)
            logger.info("Image2Image Result '%s'", response.dict())
            json_data = json.dumps(response.dict()).encode('utf-8')
            mq = MqSupporter()
            mq.createProducer(config["queue"]["topic-i2i-result"], json_data, msg.properties())
            mq.createProducer(f"{config['queue']['topic-web-img-result']}-{msg.properties()['userId']}", json_data,
                              msg.properties())
        except:
            logger.error("Image file download fail", exc_info=True)


class TaskListener(threading.Thread):
    def run(self):
        config = ExtraConfig().get_config()
        mq = MqSupporter()
        mq.createConsumer(config["queue"]["topics"], config["queue"]["subscription"], config["queue"]["consumer-name"],
                          taskHandler)
        mq.closeClient()
