from torch.utils.checkpoint import checkpoint
from modules.shared import opts

def BasicTransformerBlock_forward(self, x, context=None):
    # CLIP guidance does not support checkpointing due to the use of torch.autograd.grad()
    if opts.clip_guidance:
        return self._forward(x, context)
    else:
        return checkpoint(self._forward, x, context)

def AttentionBlock_forward(self, x):
    if opts.clip_guidance:
        return self._forward(x)
    else:
        return checkpoint(self._forward, x)

def ResBlock_forward(self, x, emb):
    if opts.clip_guidance:
        return self._forward(x, emb)
    else:
        return checkpoint(self._forward, x, emb)
