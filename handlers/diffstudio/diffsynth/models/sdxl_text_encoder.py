import torch
from .sd_text_encoder import CLIPEncoderLayer
    

class SDXLTextEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, vocab_size=49408, max_position_embeddings=77, num_encoder_layers=11, encoder_intermediate_size=3072):
        super().__init__()

        # token_embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # The text encoder is different to that in Stable Diffusion 1.x.
        # It does not include final_layer_norm.

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=1):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                break
        return embeds
    
    def state_dict_converter(self):
        return SDXLTextEncoderStateDictConverter()
    

class SDXLTextEncoder2(torch.nn.Module):
    def __init__(self, embed_dim=1280, vocab_size=49408, max_position_embeddings=77, num_encoder_layers=32, encoder_intermediate_size=5120):
        super().__init__()

        # token_embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size, num_heads=20, head_dim=64, use_quick_gelu=False) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

        # text_projection
        self.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=2):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                hidden_states = embeds
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        pooled_embeds = self.text_projection(pooled_embeds)
        return pooled_embeds, hidden_states
    
    def state_dict_converter(self):
        return SDXLTextEncoder2StateDictConverter()


class SDXLTextEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.position_embedding.weight": "position_embeds",
            "text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "text_model.final_layer_norm.bias": "final_layer_norm.bias"
        }
        attn_rename_dict = {
            "self_attn.q_proj": "attn.to_q",
            "self_attn.k_proj": "attn.to_k",
            "self_attn.v_proj": "attn.to_v",
            "self_attn.out_proj": "attn.to_out",
            "layer_norm1": "layer_norm1",
            "layer_norm2": "layer_norm2",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": "position_embeds",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": "encoders.0.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight": "encoders.0.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm2.bias": "encoders.0.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm2.weight": "encoders.0.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.bias": "encoders.0.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight": "encoders.0.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc2.bias": "encoders.0.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc2.weight": "encoders.0.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias": "encoders.0.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight": "encoders.0.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias": "encoders.0.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight": "encoders.0.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias": "encoders.0.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": "encoders.0.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias": "encoders.0.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight": "encoders.0.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.layer_norm1.bias": "encoders.1.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.layer_norm1.weight": "encoders.1.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.layer_norm2.bias": "encoders.1.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.layer_norm2.weight": "encoders.1.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.mlp.fc1.bias": "encoders.1.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.mlp.fc1.weight": "encoders.1.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.mlp.fc2.bias": "encoders.1.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.mlp.fc2.weight": "encoders.1.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.k_proj.bias": "encoders.1.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.k_proj.weight": "encoders.1.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.out_proj.bias": "encoders.1.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.out_proj.weight": "encoders.1.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.q_proj.bias": "encoders.1.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.q_proj.weight": "encoders.1.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.v_proj.bias": "encoders.1.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.1.self_attn.v_proj.weight": "encoders.1.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.layer_norm1.bias": "encoders.10.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.layer_norm1.weight": "encoders.10.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.layer_norm2.bias": "encoders.10.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.layer_norm2.weight": "encoders.10.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.mlp.fc1.bias": "encoders.10.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.mlp.fc1.weight": "encoders.10.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.mlp.fc2.bias": "encoders.10.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.mlp.fc2.weight": "encoders.10.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.k_proj.bias": "encoders.10.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.k_proj.weight": "encoders.10.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.out_proj.bias": "encoders.10.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.out_proj.weight": "encoders.10.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.q_proj.bias": "encoders.10.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight": "encoders.10.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.v_proj.bias": "encoders.10.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight": "encoders.10.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.layer_norm1.bias": "encoders.2.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.layer_norm1.weight": "encoders.2.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.layer_norm2.bias": "encoders.2.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.layer_norm2.weight": "encoders.2.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.mlp.fc1.bias": "encoders.2.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.mlp.fc1.weight": "encoders.2.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.mlp.fc2.bias": "encoders.2.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.mlp.fc2.weight": "encoders.2.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.k_proj.bias": "encoders.2.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.k_proj.weight": "encoders.2.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.out_proj.bias": "encoders.2.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.out_proj.weight": "encoders.2.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias": "encoders.2.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight": "encoders.2.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.v_proj.bias": "encoders.2.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.v_proj.weight": "encoders.2.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.layer_norm1.bias": "encoders.3.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.layer_norm1.weight": "encoders.3.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.layer_norm2.bias": "encoders.3.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.layer_norm2.weight": "encoders.3.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.mlp.fc1.bias": "encoders.3.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.mlp.fc1.weight": "encoders.3.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.mlp.fc2.bias": "encoders.3.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.mlp.fc2.weight": "encoders.3.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.k_proj.bias": "encoders.3.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.k_proj.weight": "encoders.3.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.out_proj.bias": "encoders.3.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.out_proj.weight": "encoders.3.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.q_proj.bias": "encoders.3.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight": "encoders.3.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.v_proj.bias": "encoders.3.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.3.self_attn.v_proj.weight": "encoders.3.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.layer_norm1.bias": "encoders.4.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.layer_norm1.weight": "encoders.4.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.layer_norm2.bias": "encoders.4.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.layer_norm2.weight": "encoders.4.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.mlp.fc1.bias": "encoders.4.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.mlp.fc1.weight": "encoders.4.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.mlp.fc2.bias": "encoders.4.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.mlp.fc2.weight": "encoders.4.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.k_proj.bias": "encoders.4.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.k_proj.weight": "encoders.4.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.out_proj.bias": "encoders.4.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.out_proj.weight": "encoders.4.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.q_proj.bias": "encoders.4.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.q_proj.weight": "encoders.4.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.v_proj.bias": "encoders.4.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.4.self_attn.v_proj.weight": "encoders.4.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.layer_norm1.bias": "encoders.5.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.layer_norm1.weight": "encoders.5.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.layer_norm2.bias": "encoders.5.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.layer_norm2.weight": "encoders.5.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.mlp.fc1.bias": "encoders.5.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.mlp.fc1.weight": "encoders.5.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.mlp.fc2.bias": "encoders.5.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.mlp.fc2.weight": "encoders.5.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.k_proj.bias": "encoders.5.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight": "encoders.5.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.out_proj.bias": "encoders.5.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.out_proj.weight": "encoders.5.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.q_proj.bias": "encoders.5.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight": "encoders.5.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.v_proj.bias": "encoders.5.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.v_proj.weight": "encoders.5.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.layer_norm1.bias": "encoders.6.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.layer_norm1.weight": "encoders.6.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.layer_norm2.bias": "encoders.6.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.layer_norm2.weight": "encoders.6.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.mlp.fc1.bias": "encoders.6.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.mlp.fc1.weight": "encoders.6.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.mlp.fc2.bias": "encoders.6.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.mlp.fc2.weight": "encoders.6.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.k_proj.bias": "encoders.6.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.k_proj.weight": "encoders.6.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.out_proj.bias": "encoders.6.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.out_proj.weight": "encoders.6.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.q_proj.bias": "encoders.6.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.q_proj.weight": "encoders.6.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.v_proj.bias": "encoders.6.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.6.self_attn.v_proj.weight": "encoders.6.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.layer_norm1.bias": "encoders.7.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.layer_norm1.weight": "encoders.7.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.layer_norm2.bias": "encoders.7.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.layer_norm2.weight": "encoders.7.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.mlp.fc1.bias": "encoders.7.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.mlp.fc1.weight": "encoders.7.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.mlp.fc2.bias": "encoders.7.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.mlp.fc2.weight": "encoders.7.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.k_proj.bias": "encoders.7.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.k_proj.weight": "encoders.7.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.out_proj.bias": "encoders.7.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.out_proj.weight": "encoders.7.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.q_proj.bias": "encoders.7.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.q_proj.weight": "encoders.7.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.v_proj.bias": "encoders.7.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.7.self_attn.v_proj.weight": "encoders.7.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.layer_norm1.bias": "encoders.8.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.layer_norm1.weight": "encoders.8.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.layer_norm2.bias": "encoders.8.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.layer_norm2.weight": "encoders.8.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.mlp.fc1.bias": "encoders.8.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.mlp.fc1.weight": "encoders.8.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.mlp.fc2.bias": "encoders.8.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.mlp.fc2.weight": "encoders.8.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.k_proj.bias": "encoders.8.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.k_proj.weight": "encoders.8.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.out_proj.bias": "encoders.8.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.out_proj.weight": "encoders.8.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.q_proj.bias": "encoders.8.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.q_proj.weight": "encoders.8.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.v_proj.bias": "encoders.8.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.8.self_attn.v_proj.weight": "encoders.8.attn.to_v.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.layer_norm1.bias": "encoders.9.layer_norm1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.layer_norm1.weight": "encoders.9.layer_norm1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.layer_norm2.bias": "encoders.9.layer_norm2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.layer_norm2.weight": "encoders.9.layer_norm2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.mlp.fc1.bias": "encoders.9.fc1.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.mlp.fc1.weight": "encoders.9.fc1.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.mlp.fc2.bias": "encoders.9.fc2.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.mlp.fc2.weight": "encoders.9.fc2.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.k_proj.bias": "encoders.9.attn.to_k.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.k_proj.weight": "encoders.9.attn.to_k.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.out_proj.bias": "encoders.9.attn.to_out.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.out_proj.weight": "encoders.9.attn.to_out.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.q_proj.bias": "encoders.9.attn.to_q.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.q_proj.weight": "encoders.9.attn.to_q.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.v_proj.bias": "encoders.9.attn.to_v.bias",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight": "encoders.9.attn.to_v.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
        return state_dict_


class SDXLTextEncoder2StateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.position_embedding.weight": "position_embeds",
            "text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "text_model.final_layer_norm.bias": "final_layer_norm.bias",
            "text_projection.weight": "text_projection.weight"
        }
        attn_rename_dict = {
            "self_attn.q_proj": "attn.to_q",
            "self_attn.k_proj": "attn.to_k",
            "self_attn.v_proj": "attn.to_v",
            "self_attn.out_proj": "attn.to_out",
            "layer_norm1": "layer_norm1",
            "layer_norm2": "layer_norm2",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "conditioner.embedders.1.model.ln_final.bias": "final_layer_norm.bias",
            "conditioner.embedders.1.model.ln_final.weight": "final_layer_norm.weight",
            "conditioner.embedders.1.model.positional_embedding": "position_embeds",
            "conditioner.embedders.1.model.token_embedding.weight": "token_embedding.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_bias": ['encoders.0.attn.to_q.bias', 'encoders.0.attn.to_k.bias', 'encoders.0.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight": ['encoders.0.attn.to_q.weight', 'encoders.0.attn.to_k.weight', 'encoders.0.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.bias": "encoders.0.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight": "encoders.0.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": "encoders.0.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight": "encoders.0.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_2.bias": "encoders.0.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_2.weight": "encoders.0.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc.bias": "encoders.0.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc.weight": "encoders.0.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_proj.bias": "encoders.0.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_proj.weight": "encoders.0.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.1.attn.in_proj_bias": ['encoders.1.attn.to_q.bias', 'encoders.1.attn.to_k.bias', 'encoders.1.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.1.attn.in_proj_weight": ['encoders.1.attn.to_q.weight', 'encoders.1.attn.to_k.weight', 'encoders.1.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.1.attn.out_proj.bias": "encoders.1.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.1.attn.out_proj.weight": "encoders.1.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.1.ln_1.bias": "encoders.1.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.1.ln_1.weight": "encoders.1.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.1.ln_2.bias": "encoders.1.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.1.ln_2.weight": "encoders.1.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.1.mlp.c_fc.bias": "encoders.1.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.1.mlp.c_fc.weight": "encoders.1.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.1.mlp.c_proj.bias": "encoders.1.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.1.mlp.c_proj.weight": "encoders.1.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.10.attn.in_proj_bias": ['encoders.10.attn.to_q.bias', 'encoders.10.attn.to_k.bias', 'encoders.10.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.10.attn.in_proj_weight": ['encoders.10.attn.to_q.weight', 'encoders.10.attn.to_k.weight', 'encoders.10.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.10.attn.out_proj.bias": "encoders.10.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.10.attn.out_proj.weight": "encoders.10.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.10.ln_1.bias": "encoders.10.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.10.ln_1.weight": "encoders.10.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.10.ln_2.bias": "encoders.10.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.10.ln_2.weight": "encoders.10.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.10.mlp.c_fc.bias": "encoders.10.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.10.mlp.c_fc.weight": "encoders.10.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.10.mlp.c_proj.bias": "encoders.10.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.10.mlp.c_proj.weight": "encoders.10.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.11.attn.in_proj_bias": ['encoders.11.attn.to_q.bias', 'encoders.11.attn.to_k.bias', 'encoders.11.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.11.attn.in_proj_weight": ['encoders.11.attn.to_q.weight', 'encoders.11.attn.to_k.weight', 'encoders.11.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.11.attn.out_proj.bias": "encoders.11.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.11.attn.out_proj.weight": "encoders.11.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.11.ln_1.bias": "encoders.11.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.11.ln_1.weight": "encoders.11.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.11.ln_2.bias": "encoders.11.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.11.ln_2.weight": "encoders.11.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.11.mlp.c_fc.bias": "encoders.11.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.11.mlp.c_fc.weight": "encoders.11.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.11.mlp.c_proj.bias": "encoders.11.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.11.mlp.c_proj.weight": "encoders.11.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.12.attn.in_proj_bias": ['encoders.12.attn.to_q.bias', 'encoders.12.attn.to_k.bias', 'encoders.12.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.12.attn.in_proj_weight": ['encoders.12.attn.to_q.weight', 'encoders.12.attn.to_k.weight', 'encoders.12.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.12.attn.out_proj.bias": "encoders.12.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.12.attn.out_proj.weight": "encoders.12.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.12.ln_1.bias": "encoders.12.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.12.ln_1.weight": "encoders.12.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.12.ln_2.bias": "encoders.12.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.12.ln_2.weight": "encoders.12.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.12.mlp.c_fc.bias": "encoders.12.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.12.mlp.c_fc.weight": "encoders.12.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.12.mlp.c_proj.bias": "encoders.12.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.12.mlp.c_proj.weight": "encoders.12.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.13.attn.in_proj_bias": ['encoders.13.attn.to_q.bias', 'encoders.13.attn.to_k.bias', 'encoders.13.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.13.attn.in_proj_weight": ['encoders.13.attn.to_q.weight', 'encoders.13.attn.to_k.weight', 'encoders.13.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.13.attn.out_proj.bias": "encoders.13.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.13.attn.out_proj.weight": "encoders.13.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.13.ln_1.bias": "encoders.13.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.13.ln_1.weight": "encoders.13.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.13.ln_2.bias": "encoders.13.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.13.ln_2.weight": "encoders.13.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.13.mlp.c_fc.bias": "encoders.13.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.13.mlp.c_fc.weight": "encoders.13.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.13.mlp.c_proj.bias": "encoders.13.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.13.mlp.c_proj.weight": "encoders.13.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.14.attn.in_proj_bias": ['encoders.14.attn.to_q.bias', 'encoders.14.attn.to_k.bias', 'encoders.14.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.14.attn.in_proj_weight": ['encoders.14.attn.to_q.weight', 'encoders.14.attn.to_k.weight', 'encoders.14.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.14.attn.out_proj.bias": "encoders.14.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.14.attn.out_proj.weight": "encoders.14.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.14.ln_1.bias": "encoders.14.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.14.ln_1.weight": "encoders.14.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.14.ln_2.bias": "encoders.14.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.14.ln_2.weight": "encoders.14.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.14.mlp.c_fc.bias": "encoders.14.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.14.mlp.c_fc.weight": "encoders.14.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.14.mlp.c_proj.bias": "encoders.14.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.14.mlp.c_proj.weight": "encoders.14.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.15.attn.in_proj_bias": ['encoders.15.attn.to_q.bias', 'encoders.15.attn.to_k.bias', 'encoders.15.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.15.attn.in_proj_weight": ['encoders.15.attn.to_q.weight', 'encoders.15.attn.to_k.weight', 'encoders.15.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.15.attn.out_proj.bias": "encoders.15.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.15.attn.out_proj.weight": "encoders.15.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.15.ln_1.bias": "encoders.15.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.15.ln_1.weight": "encoders.15.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.15.ln_2.bias": "encoders.15.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.15.ln_2.weight": "encoders.15.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.15.mlp.c_fc.bias": "encoders.15.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.15.mlp.c_fc.weight": "encoders.15.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.15.mlp.c_proj.bias": "encoders.15.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.15.mlp.c_proj.weight": "encoders.15.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.16.attn.in_proj_bias": ['encoders.16.attn.to_q.bias', 'encoders.16.attn.to_k.bias', 'encoders.16.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.16.attn.in_proj_weight": ['encoders.16.attn.to_q.weight', 'encoders.16.attn.to_k.weight', 'encoders.16.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.16.attn.out_proj.bias": "encoders.16.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.16.attn.out_proj.weight": "encoders.16.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.16.ln_1.bias": "encoders.16.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.16.ln_1.weight": "encoders.16.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.16.ln_2.bias": "encoders.16.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.16.ln_2.weight": "encoders.16.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.16.mlp.c_fc.bias": "encoders.16.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.16.mlp.c_fc.weight": "encoders.16.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.16.mlp.c_proj.bias": "encoders.16.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.16.mlp.c_proj.weight": "encoders.16.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.17.attn.in_proj_bias": ['encoders.17.attn.to_q.bias', 'encoders.17.attn.to_k.bias', 'encoders.17.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.17.attn.in_proj_weight": ['encoders.17.attn.to_q.weight', 'encoders.17.attn.to_k.weight', 'encoders.17.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.17.attn.out_proj.bias": "encoders.17.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.17.attn.out_proj.weight": "encoders.17.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.17.ln_1.bias": "encoders.17.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.17.ln_1.weight": "encoders.17.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.17.ln_2.bias": "encoders.17.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.17.ln_2.weight": "encoders.17.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.17.mlp.c_fc.bias": "encoders.17.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.17.mlp.c_fc.weight": "encoders.17.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.17.mlp.c_proj.bias": "encoders.17.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.17.mlp.c_proj.weight": "encoders.17.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.18.attn.in_proj_bias": ['encoders.18.attn.to_q.bias', 'encoders.18.attn.to_k.bias', 'encoders.18.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.18.attn.in_proj_weight": ['encoders.18.attn.to_q.weight', 'encoders.18.attn.to_k.weight', 'encoders.18.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.18.attn.out_proj.bias": "encoders.18.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.18.attn.out_proj.weight": "encoders.18.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.18.ln_1.bias": "encoders.18.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.18.ln_1.weight": "encoders.18.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.18.ln_2.bias": "encoders.18.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.18.ln_2.weight": "encoders.18.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.18.mlp.c_fc.bias": "encoders.18.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.18.mlp.c_fc.weight": "encoders.18.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.18.mlp.c_proj.bias": "encoders.18.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.18.mlp.c_proj.weight": "encoders.18.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.19.attn.in_proj_bias": ['encoders.19.attn.to_q.bias', 'encoders.19.attn.to_k.bias', 'encoders.19.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.19.attn.in_proj_weight": ['encoders.19.attn.to_q.weight', 'encoders.19.attn.to_k.weight', 'encoders.19.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.19.attn.out_proj.bias": "encoders.19.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.19.attn.out_proj.weight": "encoders.19.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.19.ln_1.bias": "encoders.19.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.19.ln_1.weight": "encoders.19.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.19.ln_2.bias": "encoders.19.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.19.ln_2.weight": "encoders.19.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.19.mlp.c_fc.bias": "encoders.19.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.19.mlp.c_fc.weight": "encoders.19.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.19.mlp.c_proj.bias": "encoders.19.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.19.mlp.c_proj.weight": "encoders.19.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.2.attn.in_proj_bias": ['encoders.2.attn.to_q.bias', 'encoders.2.attn.to_k.bias', 'encoders.2.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.2.attn.in_proj_weight": ['encoders.2.attn.to_q.weight', 'encoders.2.attn.to_k.weight', 'encoders.2.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.2.attn.out_proj.bias": "encoders.2.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.2.attn.out_proj.weight": "encoders.2.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.2.ln_1.bias": "encoders.2.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.2.ln_1.weight": "encoders.2.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.2.ln_2.bias": "encoders.2.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.2.ln_2.weight": "encoders.2.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.2.mlp.c_fc.bias": "encoders.2.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.2.mlp.c_fc.weight": "encoders.2.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.2.mlp.c_proj.bias": "encoders.2.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.2.mlp.c_proj.weight": "encoders.2.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.20.attn.in_proj_bias": ['encoders.20.attn.to_q.bias', 'encoders.20.attn.to_k.bias', 'encoders.20.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.20.attn.in_proj_weight": ['encoders.20.attn.to_q.weight', 'encoders.20.attn.to_k.weight', 'encoders.20.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.20.attn.out_proj.bias": "encoders.20.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.20.attn.out_proj.weight": "encoders.20.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.20.ln_1.bias": "encoders.20.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.20.ln_1.weight": "encoders.20.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.20.ln_2.bias": "encoders.20.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.20.ln_2.weight": "encoders.20.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.20.mlp.c_fc.bias": "encoders.20.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.20.mlp.c_fc.weight": "encoders.20.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.20.mlp.c_proj.bias": "encoders.20.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.20.mlp.c_proj.weight": "encoders.20.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.21.attn.in_proj_bias": ['encoders.21.attn.to_q.bias', 'encoders.21.attn.to_k.bias', 'encoders.21.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.21.attn.in_proj_weight": ['encoders.21.attn.to_q.weight', 'encoders.21.attn.to_k.weight', 'encoders.21.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.21.attn.out_proj.bias": "encoders.21.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.21.attn.out_proj.weight": "encoders.21.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.21.ln_1.bias": "encoders.21.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.21.ln_1.weight": "encoders.21.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.21.ln_2.bias": "encoders.21.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.21.ln_2.weight": "encoders.21.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.21.mlp.c_fc.bias": "encoders.21.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.21.mlp.c_fc.weight": "encoders.21.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.21.mlp.c_proj.bias": "encoders.21.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.21.mlp.c_proj.weight": "encoders.21.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.22.attn.in_proj_bias": ['encoders.22.attn.to_q.bias', 'encoders.22.attn.to_k.bias', 'encoders.22.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.22.attn.in_proj_weight": ['encoders.22.attn.to_q.weight', 'encoders.22.attn.to_k.weight', 'encoders.22.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.22.attn.out_proj.bias": "encoders.22.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.22.attn.out_proj.weight": "encoders.22.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.22.ln_1.bias": "encoders.22.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.22.ln_1.weight": "encoders.22.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.22.ln_2.bias": "encoders.22.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.22.ln_2.weight": "encoders.22.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.22.mlp.c_fc.bias": "encoders.22.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.22.mlp.c_fc.weight": "encoders.22.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.22.mlp.c_proj.bias": "encoders.22.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.22.mlp.c_proj.weight": "encoders.22.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.23.attn.in_proj_bias": ['encoders.23.attn.to_q.bias', 'encoders.23.attn.to_k.bias', 'encoders.23.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.23.attn.in_proj_weight": ['encoders.23.attn.to_q.weight', 'encoders.23.attn.to_k.weight', 'encoders.23.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.23.attn.out_proj.bias": "encoders.23.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.23.attn.out_proj.weight": "encoders.23.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.23.ln_1.bias": "encoders.23.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.23.ln_1.weight": "encoders.23.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.23.ln_2.bias": "encoders.23.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.23.ln_2.weight": "encoders.23.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.23.mlp.c_fc.bias": "encoders.23.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.23.mlp.c_fc.weight": "encoders.23.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.23.mlp.c_proj.bias": "encoders.23.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.23.mlp.c_proj.weight": "encoders.23.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.24.attn.in_proj_bias": ['encoders.24.attn.to_q.bias', 'encoders.24.attn.to_k.bias', 'encoders.24.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.24.attn.in_proj_weight": ['encoders.24.attn.to_q.weight', 'encoders.24.attn.to_k.weight', 'encoders.24.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.24.attn.out_proj.bias": "encoders.24.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.24.attn.out_proj.weight": "encoders.24.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.24.ln_1.bias": "encoders.24.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.24.ln_1.weight": "encoders.24.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.24.ln_2.bias": "encoders.24.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.24.ln_2.weight": "encoders.24.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.24.mlp.c_fc.bias": "encoders.24.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.24.mlp.c_fc.weight": "encoders.24.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.24.mlp.c_proj.bias": "encoders.24.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.24.mlp.c_proj.weight": "encoders.24.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.25.attn.in_proj_bias": ['encoders.25.attn.to_q.bias', 'encoders.25.attn.to_k.bias', 'encoders.25.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.25.attn.in_proj_weight": ['encoders.25.attn.to_q.weight', 'encoders.25.attn.to_k.weight', 'encoders.25.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.25.attn.out_proj.bias": "encoders.25.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.25.attn.out_proj.weight": "encoders.25.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.25.ln_1.bias": "encoders.25.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.25.ln_1.weight": "encoders.25.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.25.ln_2.bias": "encoders.25.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.25.ln_2.weight": "encoders.25.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.25.mlp.c_fc.bias": "encoders.25.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.25.mlp.c_fc.weight": "encoders.25.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.25.mlp.c_proj.bias": "encoders.25.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.25.mlp.c_proj.weight": "encoders.25.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.26.attn.in_proj_bias": ['encoders.26.attn.to_q.bias', 'encoders.26.attn.to_k.bias', 'encoders.26.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.26.attn.in_proj_weight": ['encoders.26.attn.to_q.weight', 'encoders.26.attn.to_k.weight', 'encoders.26.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.26.attn.out_proj.bias": "encoders.26.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.26.attn.out_proj.weight": "encoders.26.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.26.ln_1.bias": "encoders.26.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.26.ln_1.weight": "encoders.26.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.26.ln_2.bias": "encoders.26.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.26.ln_2.weight": "encoders.26.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.26.mlp.c_fc.bias": "encoders.26.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.26.mlp.c_fc.weight": "encoders.26.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.26.mlp.c_proj.bias": "encoders.26.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.26.mlp.c_proj.weight": "encoders.26.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.27.attn.in_proj_bias": ['encoders.27.attn.to_q.bias', 'encoders.27.attn.to_k.bias', 'encoders.27.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.27.attn.in_proj_weight": ['encoders.27.attn.to_q.weight', 'encoders.27.attn.to_k.weight', 'encoders.27.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.27.attn.out_proj.bias": "encoders.27.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.27.attn.out_proj.weight": "encoders.27.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.27.ln_1.bias": "encoders.27.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.27.ln_1.weight": "encoders.27.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.27.ln_2.bias": "encoders.27.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.27.ln_2.weight": "encoders.27.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.27.mlp.c_fc.bias": "encoders.27.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.27.mlp.c_fc.weight": "encoders.27.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.27.mlp.c_proj.bias": "encoders.27.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.27.mlp.c_proj.weight": "encoders.27.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.28.attn.in_proj_bias": ['encoders.28.attn.to_q.bias', 'encoders.28.attn.to_k.bias', 'encoders.28.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.28.attn.in_proj_weight": ['encoders.28.attn.to_q.weight', 'encoders.28.attn.to_k.weight', 'encoders.28.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.28.attn.out_proj.bias": "encoders.28.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.28.attn.out_proj.weight": "encoders.28.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.28.ln_1.bias": "encoders.28.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.28.ln_1.weight": "encoders.28.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.28.ln_2.bias": "encoders.28.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.28.ln_2.weight": "encoders.28.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.28.mlp.c_fc.bias": "encoders.28.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.28.mlp.c_fc.weight": "encoders.28.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.28.mlp.c_proj.bias": "encoders.28.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.28.mlp.c_proj.weight": "encoders.28.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.29.attn.in_proj_bias": ['encoders.29.attn.to_q.bias', 'encoders.29.attn.to_k.bias', 'encoders.29.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.29.attn.in_proj_weight": ['encoders.29.attn.to_q.weight', 'encoders.29.attn.to_k.weight', 'encoders.29.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.29.attn.out_proj.bias": "encoders.29.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.29.attn.out_proj.weight": "encoders.29.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.29.ln_1.bias": "encoders.29.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.29.ln_1.weight": "encoders.29.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.29.ln_2.bias": "encoders.29.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.29.ln_2.weight": "encoders.29.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.29.mlp.c_fc.bias": "encoders.29.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.29.mlp.c_fc.weight": "encoders.29.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.29.mlp.c_proj.bias": "encoders.29.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.29.mlp.c_proj.weight": "encoders.29.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.3.attn.in_proj_bias": ['encoders.3.attn.to_q.bias', 'encoders.3.attn.to_k.bias', 'encoders.3.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.3.attn.in_proj_weight": ['encoders.3.attn.to_q.weight', 'encoders.3.attn.to_k.weight', 'encoders.3.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.3.attn.out_proj.bias": "encoders.3.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.3.attn.out_proj.weight": "encoders.3.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.3.ln_1.bias": "encoders.3.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.3.ln_1.weight": "encoders.3.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.3.ln_2.bias": "encoders.3.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.3.ln_2.weight": "encoders.3.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.3.mlp.c_fc.bias": "encoders.3.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.3.mlp.c_fc.weight": "encoders.3.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.3.mlp.c_proj.bias": "encoders.3.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.3.mlp.c_proj.weight": "encoders.3.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.30.attn.in_proj_bias": ['encoders.30.attn.to_q.bias', 'encoders.30.attn.to_k.bias', 'encoders.30.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.30.attn.in_proj_weight": ['encoders.30.attn.to_q.weight', 'encoders.30.attn.to_k.weight', 'encoders.30.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.30.attn.out_proj.bias": "encoders.30.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.30.attn.out_proj.weight": "encoders.30.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.30.ln_1.bias": "encoders.30.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.30.ln_1.weight": "encoders.30.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.30.ln_2.bias": "encoders.30.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.30.ln_2.weight": "encoders.30.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.30.mlp.c_fc.bias": "encoders.30.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.30.mlp.c_fc.weight": "encoders.30.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.30.mlp.c_proj.bias": "encoders.30.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.30.mlp.c_proj.weight": "encoders.30.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.31.attn.in_proj_bias": ['encoders.31.attn.to_q.bias', 'encoders.31.attn.to_k.bias', 'encoders.31.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.31.attn.in_proj_weight": ['encoders.31.attn.to_q.weight', 'encoders.31.attn.to_k.weight', 'encoders.31.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.31.attn.out_proj.bias": "encoders.31.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.31.attn.out_proj.weight": "encoders.31.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.31.ln_1.bias": "encoders.31.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.31.ln_1.weight": "encoders.31.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.31.ln_2.bias": "encoders.31.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.31.ln_2.weight": "encoders.31.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.31.mlp.c_fc.bias": "encoders.31.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.31.mlp.c_fc.weight": "encoders.31.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.31.mlp.c_proj.bias": "encoders.31.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.31.mlp.c_proj.weight": "encoders.31.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.4.attn.in_proj_bias": ['encoders.4.attn.to_q.bias', 'encoders.4.attn.to_k.bias', 'encoders.4.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.4.attn.in_proj_weight": ['encoders.4.attn.to_q.weight', 'encoders.4.attn.to_k.weight', 'encoders.4.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.4.attn.out_proj.bias": "encoders.4.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.4.attn.out_proj.weight": "encoders.4.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.4.ln_1.bias": "encoders.4.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.4.ln_1.weight": "encoders.4.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.4.ln_2.bias": "encoders.4.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.4.ln_2.weight": "encoders.4.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.4.mlp.c_fc.bias": "encoders.4.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.4.mlp.c_fc.weight": "encoders.4.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.4.mlp.c_proj.bias": "encoders.4.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.4.mlp.c_proj.weight": "encoders.4.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.5.attn.in_proj_bias": ['encoders.5.attn.to_q.bias', 'encoders.5.attn.to_k.bias', 'encoders.5.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.5.attn.in_proj_weight": ['encoders.5.attn.to_q.weight', 'encoders.5.attn.to_k.weight', 'encoders.5.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.5.attn.out_proj.bias": "encoders.5.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.5.attn.out_proj.weight": "encoders.5.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.5.ln_1.bias": "encoders.5.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.5.ln_1.weight": "encoders.5.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.5.ln_2.bias": "encoders.5.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.5.ln_2.weight": "encoders.5.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.5.mlp.c_fc.bias": "encoders.5.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.5.mlp.c_fc.weight": "encoders.5.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.5.mlp.c_proj.bias": "encoders.5.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.5.mlp.c_proj.weight": "encoders.5.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.6.attn.in_proj_bias": ['encoders.6.attn.to_q.bias', 'encoders.6.attn.to_k.bias', 'encoders.6.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.6.attn.in_proj_weight": ['encoders.6.attn.to_q.weight', 'encoders.6.attn.to_k.weight', 'encoders.6.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.6.attn.out_proj.bias": "encoders.6.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.6.attn.out_proj.weight": "encoders.6.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.6.ln_1.bias": "encoders.6.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.6.ln_1.weight": "encoders.6.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.6.ln_2.bias": "encoders.6.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.6.ln_2.weight": "encoders.6.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.6.mlp.c_fc.bias": "encoders.6.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.6.mlp.c_fc.weight": "encoders.6.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.6.mlp.c_proj.bias": "encoders.6.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.6.mlp.c_proj.weight": "encoders.6.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.7.attn.in_proj_bias": ['encoders.7.attn.to_q.bias', 'encoders.7.attn.to_k.bias', 'encoders.7.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.7.attn.in_proj_weight": ['encoders.7.attn.to_q.weight', 'encoders.7.attn.to_k.weight', 'encoders.7.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.7.attn.out_proj.bias": "encoders.7.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.7.attn.out_proj.weight": "encoders.7.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.7.ln_1.bias": "encoders.7.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.7.ln_1.weight": "encoders.7.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.7.ln_2.bias": "encoders.7.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.7.ln_2.weight": "encoders.7.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.7.mlp.c_fc.bias": "encoders.7.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.7.mlp.c_fc.weight": "encoders.7.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.7.mlp.c_proj.bias": "encoders.7.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.7.mlp.c_proj.weight": "encoders.7.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.8.attn.in_proj_bias": ['encoders.8.attn.to_q.bias', 'encoders.8.attn.to_k.bias', 'encoders.8.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.8.attn.in_proj_weight": ['encoders.8.attn.to_q.weight', 'encoders.8.attn.to_k.weight', 'encoders.8.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.8.attn.out_proj.bias": "encoders.8.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.8.attn.out_proj.weight": "encoders.8.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.8.ln_1.bias": "encoders.8.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.8.ln_1.weight": "encoders.8.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.8.ln_2.bias": "encoders.8.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.8.ln_2.weight": "encoders.8.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.8.mlp.c_fc.bias": "encoders.8.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.8.mlp.c_fc.weight": "encoders.8.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.8.mlp.c_proj.bias": "encoders.8.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.8.mlp.c_proj.weight": "encoders.8.fc2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_bias": ['encoders.9.attn.to_q.bias', 'encoders.9.attn.to_k.bias', 'encoders.9.attn.to_v.bias'],
            "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_weight": ['encoders.9.attn.to_q.weight', 'encoders.9.attn.to_k.weight', 'encoders.9.attn.to_v.weight'],
            "conditioner.embedders.1.model.transformer.resblocks.9.attn.out_proj.bias": "encoders.9.attn.to_out.bias",
            "conditioner.embedders.1.model.transformer.resblocks.9.attn.out_proj.weight": "encoders.9.attn.to_out.weight",
            "conditioner.embedders.1.model.transformer.resblocks.9.ln_1.bias": "encoders.9.layer_norm1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.9.ln_1.weight": "encoders.9.layer_norm1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.9.ln_2.bias": "encoders.9.layer_norm2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.9.ln_2.weight": "encoders.9.layer_norm2.weight",
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_fc.bias": "encoders.9.fc1.bias",
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_fc.weight": "encoders.9.fc1.weight",
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": "encoders.9.fc2.bias",
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight": "encoders.9.fc2.weight",
            "conditioner.embedders.1.model.text_projection": "text_projection.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "conditioner.embedders.1.model.positional_embedding":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                elif name == "conditioner.embedders.1.model.text_projection":
                    param = param.T
                if isinstance(rename_dict[name], str):
                    state_dict_[rename_dict[name]] = param
                else:
                    length = param.shape[0] // 3
                    for i, rename in enumerate(rename_dict[name]):
                        state_dict_[rename] = param[i*length: i*length+length]
        return state_dict_