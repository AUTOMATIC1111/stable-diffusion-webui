    """TODO: Add docstring."""
from torch.utils.checkpoint import checkpoint

import ldm.modules.attention
import ldm.modules.diffusionmodules.openaimodel


def BasicTransformerBlock_forward(self, x, context=None):
        """TODO: Add docstring."""
    return checkpoint(self._forward, x, context)


def AttentionBlock_forward(self, x):
        """TODO: Add docstring."""
    return checkpoint(self._forward, x)


def ResBlock_forward(self, x, emb):
        """TODO: Add docstring."""
    return checkpoint(self._forward, x, emb)


stored = []


def add():
        """TODO: Add docstring."""
    if len(stored) != 0:
        return

    stored.extend([
        ldm.modules.attention.BasicTransformerBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.ResBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward
    ])

    ldm.modules.attention.BasicTransformerBlock.forward = BasicTransformerBlock_forward
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = ResBlock_forward
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = AttentionBlock_forward


def remove():
        """TODO: Add docstring."""
    if len(stored) == 0:
        return

    ldm.modules.attention.BasicTransformerBlock.forward = stored[0]
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = stored[1]
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = stored[2]

    stored.clear()

