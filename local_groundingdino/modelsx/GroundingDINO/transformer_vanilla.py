# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
from torch import Tensor, nn

from .utils import (
    _get_activation_fn,
    _get_clones,
)


class TextTransformer(nn.Module):
    def __init__(self, num_layers, d_model=256, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.norm = None

        single_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.layers = _get_clones(single_encoder_layer, num_layers)

    def forward(self, memory_text: torch.Tensor, text_attention_mask: torch.Tensor):
        """

        Args:
            text_attention_mask: bs, num_token
            memory_text: bs, num_token, d_model

        Raises:
            RuntimeError: _description_

        Returns:
            output: bs, num_token, d_model
        """

        output = memory_text.transpose(0, 1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=text_attention_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output.transpose(0, 1)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
