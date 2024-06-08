from torch.utils.checkpoint import checkpoint

import ldm.modules.attention
import ldm.modules.diffusionmodules.openaimodel


# Setting flag=False so that torch skips checking parameters.
# parameters checking is expensive in frequent operations.

def BasicTransformerBlock_forward(self, x, context=None):
    return checkpoint(self._forward, x, context, flag=False)


def AttentionBlock_forward(self, x):
    return checkpoint(self._forward, x, flag=False)


def ResBlock_forward(self, x, emb):
    return checkpoint(self._forward, x, emb, flag=False)


stored = []


def add():
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
    if len(stored) == 0:
        return

    ldm.modules.attention.BasicTransformerBlock.forward = stored[0]
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = stored[1]
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = stored[2]

    stored.clear()

