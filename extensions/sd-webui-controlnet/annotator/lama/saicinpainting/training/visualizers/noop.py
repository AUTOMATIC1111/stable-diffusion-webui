from annotator.lama.saicinpainting.training.visualizers.base import BaseVisualizer


class NoopVisualizer(BaseVisualizer):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        pass
