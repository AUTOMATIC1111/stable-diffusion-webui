from torch.utils.checkpoint import checkpoint

def BasicTransformerBlock_forward(self, x, context=None):
    return checkpoint(self._forward, x, context)

def AttentionBlock_forward(self, x):
    return checkpoint(self._forward, x)

def ResBlock_forward(self, x, emb):
    return checkpoint(self._forward, x, emb)