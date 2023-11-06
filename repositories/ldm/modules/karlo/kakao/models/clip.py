# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Adapted from OpenAI's CLIP (https://github.com/openai/CLIP/)
# ------------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from clip.model import CLIP, convert_weights
from clip.simple_tokenizer import SimpleTokenizer, default_bpe


"""===== Monkey-Patching original CLIP for JIT compile ====="""


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = F.layer_norm(
            x.type(torch.float32),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return ret.type(orig_type)


clip.model.LayerNorm = LayerNorm
delattr(clip.model.CLIP, "forward")

"""===== End of Monkey-Patching ====="""


class CustomizedCLIP(CLIP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.jit.export
    def encode_image(self, image):
        return self.visual(image)

    @torch.jit.export
    def encode_text(self, text):
        # re-define this function to return unpooled text features

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x_seq = x
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_out = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x_out, x_seq

    @torch.jit.ignore
    def forward(self, image, text):
        super().forward(image, text)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu").state_dict()

        vit = "visual.proj" in state_dict
        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [
                    k
                    for k in state_dict.keys()
                    if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
                ]
            )
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round(
                (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
            )
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [
                len(
                    set(
                        k.split(".")[2]
                        for k in state_dict
                        if k.startswith(f"visual.layer{b}")
                    )
                )
                for b in [1, 2, 3, 4]
            ]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round(
                (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
            )
            vision_patch_size = None
            assert (
                output_width**2 + 1
                == state_dict["visual.attnpool.positional_embedding"].shape[0]
            )
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(
                k.split(".")[2]
                for k in state_dict
                if k.startswith("transformer.resblocks")
            )
        )

        model = cls(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        convert_weights(model)
        model.load_state_dict(state_dict)
        model.eval()
        model.float()
        return model


class CustomizedTokenizer(SimpleTokenizer):
    def __init__(self):
        super().__init__(bpe_path=default_bpe())

        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]

    def padded_tokens_and_mask(self, texts, text_ctx):
        assert isinstance(texts, list) and all(
            isinstance(elem, str) for elem in texts
        ), "texts should be a list of strings"

        all_tokens = [
            [self.sot_token] + self.encode(text) + [self.eot_token] for text in texts
        ]

        mask = [
            [True] * min(text_ctx, len(tokens))
            + [False] * max(text_ctx - len(tokens), 0)
            for tokens in all_tokens
        ]
        mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.zeros(len(all_tokens), text_ctx, dtype=torch.int)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > text_ctx:
                tokens = tokens[:text_ctx]
                tokens[-1] = self.eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result, mask
