### This file contains impls for underlying related models (CLIP, T5, etc)

import torch
import math
from torch import nn
from transformers import CLIPTokenizer, T5TokenizerFast

from modules import sd_hijack


#################################################################################################
### Core/Utility
#################################################################################################


class AutocastLinear(nn.Linear):
    """Same as usual linear layer, but casts its weights to whatever the parameter type is.

    This is different from torch.autocast in a way that float16 layer processing float32 input
    will return float16 with autocast on, and float32 with this. T5 seems to be fucked
    if you do it in full float16 (returning almost all zeros in the final output).
    """

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = [t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################################
### CLIP
#################################################################################################


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}

class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        #self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)
        self.mlp = Mlp(embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation], dtype=dtype, device=device)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, textual_inversion_key="clip_l"):
        super().__init__()
        self.token_embedding = sd_hijack.TextualInversionEmbeddings(vocab_size, embed_dim, dtype=dtype, device=device, textual_inversion_key=textual_inversion_key)
        self.position_embedding = torch.nn.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device, textual_inversion_key=config_dict.get('textual_inversion_key', 'clip_l'))
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        x = self.embeddings(input_tokens)
        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        x, i = self.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[torch.arange(x.shape[0], device=x.device), input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class SDTokenizer:
    def __init__(self, max_length=77, pad_with_end=True, tokenizer=None, has_start_token=True, pad_to_max_length=True, min_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer('')["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8


    def tokenize_with_weights(self, text:str):
        """Tokenize the text, with weight values - presume 1.0 for all and ignore other features here. The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3."""
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(' ')
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            batch.extend([(t, 1) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


class SD3Tokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text:str):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text)
        out["l"] = self.clip_l.tokenize_with_weights(text)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text)
        return out


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = [a[0] for a in token_weight_pairs[0]]
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ["last", "pooled", "hidden"]
    def __init__(self, device="cpu", max_length=77, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=CLIPTextModel,
                 special_tokens=None, layer_norm_hidden_state=True, return_projected_pooled=True):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens if special_tokens is not None else {"start": 49406, "end": 49407, "pad": 49407}
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = torch.asarray(tokens, dtype=torch.int64, device=backup_embeds.weight.device)
        outputs = self.transformer(tokens, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return z.float(), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""
    def __init__(self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None):
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={"start": 49406, "end": 49407, "pad": 0}, layer_norm_hidden_state=False)


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience"""
    def __init__(self, config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5)


#################################################################################################
### T5 implementation, for the T5-XXL text encoder portion, largely pulled from upstream impl
#################################################################################################

class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""
    def __init__(self):
        super().__init__(pad_with_end=False, tokenizer=T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"), has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=77)


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(device=x.device, dtype=x.dtype) * x


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = AutocastLinear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = AutocastLinear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = AutocastLinear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = AutocastLinear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = AutocastLinear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = AutocastLinear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = AutocastLinear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device)
        if past_bias is not None:
            mask = past_bias
        else:
            mask = None

        out = attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask.to(x.dtype) if mask is not None else None)

        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads, vocab_size, dtype, device):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim, device=device)
        self.block = torch.nn.ModuleList([T5Block(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias=(i == 0), dtype=dtype, device=device) for i in range(num_layers)])
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, input_ids, intermediate_output=None, final_layer_norm_intermediate=True):
        intermediate = None
        x = self.embed_tokens(input_ids).to(torch.float32)  # needs float32 or else T5 returns all zeroes
        past_bias = None
        for i, layer in enumerate(self.block):
            x, past_bias = layer(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        self.encoder = T5Stack(self.num_layers, config_dict["d_model"], config_dict["d_model"], config_dict["d_ff"], config_dict["num_heads"], config_dict["vocab_size"], dtype, device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
