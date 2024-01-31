from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        """Placeholder for sample function."""
