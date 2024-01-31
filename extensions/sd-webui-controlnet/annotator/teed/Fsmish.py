"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
 "Smish: A Novel Activation Function for Deep Learning Methods.
 " Electronics 11.4 (2022): 540.
"""

# import pytorch
import torch
import torch.nn.functional as F


@torch.jit.script
def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    return input * torch.tanh(torch.log(1+torch.sigmoid(input)))