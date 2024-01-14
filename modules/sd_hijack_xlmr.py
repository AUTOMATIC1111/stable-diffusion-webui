import torch

from modules import sd_hijack_clip, devices


class FrozenXLMREmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.id_start = wrapped.config.bos_token_id
        self.id_end = wrapped.config.eos_token_id
        self.id_pad = wrapped.config.pad_token_id

        self.comma_token = self.tokenizer.get_vocab().get(',', None)  # alt diffusion doesn't have </w> bits for comma

    def encode_with_transformers(self, tokens):
        # there's no CLIP Skip here because all hidden layers have size of 1024 and the last one uses a
        # trained layer to transform those 1024 into 768 for unet; so you can't choose which transformer
        # layer to work with - you have to use the last

        attention_mask = (tokens != self.id_pad).to(device=tokens.device, dtype=torch.int64)
        features = self.wrapped(input_ids=tokens, attention_mask=attention_mask)
        z = features['projection_state']

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.roberta.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)

        return embedded
