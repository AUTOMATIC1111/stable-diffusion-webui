import os
import safetensors
import torch
import typing

from transformers import CLIPTokenizer, T5TokenizerFast

from modules import shared, devices, modelloader, sd_hijack_clip, prompt_parser
from modules.models.sd3.other_impls import SDClipModel, SDXLClipG, T5XXLModel, SD3Tokenizer


class SafetensorsMapping(typing.Mapping):
    def __init__(self, file):
        self.file = file

    def __len__(self):
        return len(self.file.keys())

    def __iter__(self):
        for key in self.file.keys():
            yield key

    def __getitem__(self, key):
        return self.file.get_tensor(key)


CLIPL_URL = "https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/clip_l.safetensors"
CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

CLIPG_URL = "https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/clip_g.safetensors"
CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

T5_URL = "https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/t5xxl_fp16.safetensors"
T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class Sd3ClipLG(sd_hijack_clip.TextConditionalModel):
    def __init__(self, clip_l, clip_g):
        super().__init__()

        self.clip_l = clip_l
        self.clip_g = clip_g

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        empty = self.tokenizer('')["input_ids"]
        self.id_start = empty[0]
        self.id_end = empty[1]
        self.id_pad = empty[1]

        self.return_pooled = True

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def encode_with_transformers(self, tokens):
        tokens_g = tokens.clone()

        for batch_pos in range(tokens_g.shape[0]):
            index = tokens_g[batch_pos].cpu().tolist().index(self.id_end)
            tokens_g[batch_pos, index+1:tokens_g.shape[1]] = 0

        l_out, l_pooled = self.clip_l(tokens)
        g_out, g_pooled = self.clip_g(tokens_g)

        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))

        vector_out = torch.cat((l_pooled, g_pooled), dim=-1)

        lg_out.pooled = vector_out
        return lg_out

    def encode_embedding_init_text(self, init_text, nvpt):
        return torch.zeros((nvpt, 768+1280), device=devices.device) # XXX


class Sd3T5(torch.nn.Module):
    def __init__(self, t5xxl):
        super().__init__()

        self.t5xxl = t5xxl
        self.tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

        empty = self.tokenizer('', padding='max_length', max_length=2)["input_ids"]
        self.id_end = empty[0]
        self.id_pad = empty[1]

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def tokenize_line(self, line, *, target_token_count=None):
        if shared.opts.emphasis != "None":
            parsed = prompt_parser.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        tokens = []
        multipliers = []

        for text_tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                continue

            tokens += text_tokens
            multipliers += [weight] * len(text_tokens)

        tokens += [self.id_end]
        multipliers += [1.0]

        if target_token_count is not None:
            if len(tokens) < target_token_count:
                tokens += [self.id_pad] * (target_token_count - len(tokens))
                multipliers += [1.0] * (target_token_count - len(tokens))
            else:
                tokens = tokens[0:target_token_count]
                multipliers = multipliers[0:target_token_count]

        return tokens, multipliers

    def forward(self, texts, *, token_count):
        if not self.t5xxl or not shared.opts.sd3_enable_t5:
            return torch.zeros((len(texts), token_count, 4096), device=devices.device, dtype=devices.dtype)

        tokens_batch = []

        for text in texts:
            tokens, multipliers = self.tokenize_line(text, target_token_count=token_count)
            tokens_batch.append(tokens)

        t5_out, t5_pooled = self.t5xxl(tokens_batch)

        return t5_out

    def encode_embedding_init_text(self, init_text, nvpt):
        return torch.zeros((nvpt, 4096), device=devices.device) # XXX


class SD3Cond(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = SD3Tokenizer()

        with torch.no_grad():
            self.clip_g = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=devices.dtype)
            self.clip_l = SDClipModel(layer="hidden", layer_idx=-2, device="cpu", dtype=devices.dtype, layer_norm_hidden_state=False, return_projected_pooled=False, textmodel_json_config=CLIPL_CONFIG)

            if shared.opts.sd3_enable_t5:
                self.t5xxl = T5XXLModel(T5_CONFIG, device="cpu", dtype=devices.dtype)
            else:
                self.t5xxl = None

            self.model_lg = Sd3ClipLG(self.clip_l, self.clip_g)
            self.model_t5 = Sd3T5(self.t5xxl)

    def forward(self, prompts: list[str]):
        with devices.without_autocast():
            lg_out, vector_out = self.model_lg(prompts)
            t5_out = self.model_t5(prompts, token_count=lg_out.shape[1])
            lgt_out = torch.cat([lg_out, t5_out], dim=-2)

        return {
            'crossattn': lgt_out,
            'vector': vector_out,
        }

    def before_load_weights(self, state_dict):
        clip_path = os.path.join(shared.models_path, "CLIP")

        if 'text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight' not in state_dict:
            clip_g_file = modelloader.load_file_from_url(CLIPG_URL, model_dir=clip_path, file_name="clip_g.safetensors")
            with safetensors.safe_open(clip_g_file, framework="pt") as file:
                self.clip_g.transformer.load_state_dict(SafetensorsMapping(file))

        if 'text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight' not in state_dict:
            clip_l_file = modelloader.load_file_from_url(CLIPL_URL, model_dir=clip_path, file_name="clip_l.safetensors")
            with safetensors.safe_open(clip_l_file, framework="pt") as file:
                self.clip_l.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

        if self.t5xxl and 'text_encoders.t5xxl.transformer.encoder.embed_tokens.weight' not in state_dict:
            t5_file = modelloader.load_file_from_url(T5_URL, model_dir=clip_path, file_name="t5xxl_fp16.safetensors")
            with safetensors.safe_open(t5_file, framework="pt") as file:
                self.t5xxl.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

    def encode_embedding_init_text(self, init_text, nvpt):
        return torch.tensor([[0]], device=devices.device) # XXX

    def medvram_modules(self):
        return [self.clip_g, self.clip_l, self.t5xxl]

    def get_token_count(self, text):
        _, token_count = self.model_lg.process_texts([text])

        return token_count

    def get_target_prompt_token_count(self, token_count):
        return self.model_lg.get_target_prompt_token_count(token_count)
