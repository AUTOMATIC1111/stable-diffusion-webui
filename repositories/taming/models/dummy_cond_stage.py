from torch import Tensor


class DummyCondStage:
    def __init__(self, conditional_key):
        self.conditional_key = conditional_key
        self.train = None

    def eval(self):
        return self

    @staticmethod
    def encode(c: Tensor):
        return c, None, (None, None, c)

    @staticmethod
    def decode(c: Tensor):
        return c

    @staticmethod
    def to_rgb(c: Tensor):
        return c
