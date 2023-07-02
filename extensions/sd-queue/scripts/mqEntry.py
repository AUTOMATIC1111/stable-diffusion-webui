import threading

import pulsar
from _pulsar import ConsumerType

from scripts.extLogging import logger
from extra.loadYamlFile import ExtraConfig


class MqSupporter:
    def __init__(self, environment=None):
        self.client = None
        self.environment = environment

    def initClient(self):
        config = ExtraConfig(self.environment).get_config()
        self.client = pulsar.Client(config["queue"]["server-addr"])

    def createProducer(self, topic, msg, properties=None):
        self.initClient()
        producer = self.client.create_producer(topic)
        producer.send(msg, properties)
        self.closeClient()

    def createConsumer(self, topics: str, subscription: str, consumerName: str, messageHandler, environment):
        self.initClient()
        consumer = self.client.subscribe(topics, subscription, consumer_type=ConsumerType.Shared,
                                         consumer_name=consumerName)

        while True:
            msg = consumer.receive()
            try:
                logger.info("Received message '%s' id='%s'", msg.data(), msg.message_id())
                messageHandler(msg, environment)
                logger.info("Received message handler success")
                # Acknowledge successful processing of the message
                consumer.acknowledge(msg)
            except threading.ThreadError as e:
                logger.info("Redo task to queue, the error [%s]", e)
                consumer.negative_acknowledge(msg)
            except:
                # Message failed to be processed
                logger.error("Received message handler failure", exc_info=True)
                # consumer.negative_acknowledge(msg)
                consumer.acknowledge(msg)

    def closeClient(self):
        self.client.close()


class QueueException(Exception):
    pass
