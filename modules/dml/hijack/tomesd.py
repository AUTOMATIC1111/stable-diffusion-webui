import torch
import tomesd
from typing import Type
from modules.dml.hijack.utils import catch_nan

def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = tomesd.patch.compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = catch_nan(lambda: (u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x))
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock
tomesd.patch.make_tome_block = make_tome_block
