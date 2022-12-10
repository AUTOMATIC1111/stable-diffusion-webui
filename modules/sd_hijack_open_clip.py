import open_clip.tokenizer
import torch

from modules import sd_hijack_clip, devices
from modules.shared import opts

tokenizer = open_clip.tokenizer._tokenizer


class FrozenOpenCLIPEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        self.id_start = tokenizer.encoder["<start_of_text>"]
        self.id_end = tokenizer.encoder["<end_of_text>"]
        self.id_pad = 0

    def tokenize(self, texts):
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'

        tokenized = [tokenizer.encode(text) for text in texts]

        return tokenized

    def encode_with_transformers(self, tokens):
        # set self.wrapped.layer_idx here according to opts.CLIP_stop_at_last_layers
        z = self.wrapped.encode_with_transformer(tokens)

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        ids = tokenizer.encode(init_text)
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        embedded = self.wrapped.model.token_embedding.wrapped(ids).squeeze(0)

        return embedded
