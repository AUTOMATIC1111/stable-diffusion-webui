# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4, use_class_token=False):
        """ViT-like transformer block

        Args:
            in_channels (int): Input channels
            patch_size (int, optional): patch size. Defaults to 10.
            embedding_dim (int, optional): Embedding dimension in transformer model. Defaults to 128.
            num_heads (int, optional): number of attention heads. Defaults to 4.
            use_class_token (bool, optional): Whether to use extra token at the start for global accumulation (called as "class token"). Defaults to False.
        """
        super(PatchTransformerEncoder, self).__init__()
        self.use_class_token = use_class_token
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        
    def positional_encoding_1d(self, sequence_length, batch_size, embedding_dim, device='cpu'):
        """Generate positional encodings

        Args:
            sequence_length (int): Sequence length
            embedding_dim (int): Embedding dimension

        Returns:
            torch.Tensor SBE: Positional encodings
        """
        position = torch.arange(
            0, sequence_length, dtype=torch.float32, device=device).unsqueeze(1)
        index = torch.arange(
            0, embedding_dim, 2, dtype=torch.float32, device=device).unsqueeze(0)
        div_term = torch.exp(index * (-torch.log(torch.tensor(10000.0, device=device)) / embedding_dim))
        pos_encoding = position * div_term
        pos_encoding = torch.cat([torch.sin(pos_encoding), torch.cos(pos_encoding)], dim=1)
        pos_encoding = pos_encoding.unsqueeze(1).repeat(1, batch_size, 1)
        return pos_encoding
        

    def forward(self, x):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Input feature tensor

        Returns:
            torch.Tensor - SNE: Transformer output embeddings. S - sequence length (=HW/patch_size^2), N - batch size, E - embedding dim
        """
        embeddings = self.embedding_convPxP(x).flatten(
            2)  # .shape = n,c,s = n, embedding_dim, s
        if self.use_class_token:
            # extra special token at start ?
            embeddings = nn.functional.pad(embeddings, (1, 0))
        
        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        S, N, E = embeddings.shape
        embeddings = embeddings + self.positional_encoding_1d(S, N, E, device=embeddings.device)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x
