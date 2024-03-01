import torch
from .attention import Attention


class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads=12, head_dim=64, use_quick_gelu=True):
        super().__init__()
        self.attn = Attention(q_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, embed_dim)

        self.use_quick_gelu = use_quick_gelu

    def quickGELU(self, x):
        return x * torch.sigmoid(1.702 * x)
    
    def forward(self, hidden_states, attn_mask):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.use_quick_gelu:
            hidden_states = self.quickGELU(hidden_states)
        else:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    

class SDTextEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, vocab_size=49408, max_position_embeddings=77, num_encoder_layers=12, encoder_intermediate_size=3072):
        super().__init__()

        # token_embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

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
        embeds = self.final_layer_norm(embeds)
        return embeds
    
    def state_dict_converter(self):
        return SDTextEncoderStateDictConverter()


class SDTextEncoderStateDictConverter:
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
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": "encoders.0.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight": "encoders.0.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.bias": "encoders.0.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.weight": "encoders.0.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.bias": "encoders.0.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight": "encoders.0.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.bias": "encoders.0.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight": "encoders.0.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias": "encoders.0.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight": "encoders.0.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias": "encoders.0.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight": "encoders.0.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias": "encoders.0.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": "encoders.0.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias": "encoders.0.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight": "encoders.0.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.bias": "encoders.1.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.weight": "encoders.1.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.bias": "encoders.1.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.weight": "encoders.1.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.bias": "encoders.1.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.weight": "encoders.1.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.bias": "encoders.1.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight": "encoders.1.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.bias": "encoders.1.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.weight": "encoders.1.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.bias": "encoders.1.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.weight": "encoders.1.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.bias": "encoders.1.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.weight": "encoders.1.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.bias": "encoders.1.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.weight": "encoders.1.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.bias": "encoders.10.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.weight": "encoders.10.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.bias": "encoders.10.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.weight": "encoders.10.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.bias": "encoders.10.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.weight": "encoders.10.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.bias": "encoders.10.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.weight": "encoders.10.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.bias": "encoders.10.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.weight": "encoders.10.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.bias": "encoders.10.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.weight": "encoders.10.attn.to_out.weight",        
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.bias": "encoders.10.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight": "encoders.10.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.bias": "encoders.10.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight": "encoders.10.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.bias": "encoders.11.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.weight": "encoders.11.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.bias": "encoders.11.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.weight": "encoders.11.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.bias": "encoders.11.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.weight": "encoders.11.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.bias": "encoders.11.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.weight": "encoders.11.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.bias": "encoders.11.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.weight": "encoders.11.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.bias": "encoders.11.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.weight": "encoders.11.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.bias": "encoders.11.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight": "encoders.11.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.bias": "encoders.11.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.weight": "encoders.11.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.bias": "encoders.2.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.weight": "encoders.2.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.bias": "encoders.2.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.weight": "encoders.2.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.bias": "encoders.2.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.weight": "encoders.2.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.bias": "encoders.2.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.weight": "encoders.2.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.bias": "encoders.2.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.weight": "encoders.2.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.bias": "encoders.2.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.weight": "encoders.2.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias": "encoders.2.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight": "encoders.2.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.bias": "encoders.2.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.weight": "encoders.2.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.bias": "encoders.3.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.weight": "encoders.3.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.bias": "encoders.3.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.weight": "encoders.3.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.bias": "encoders.3.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight": "encoders.3.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.bias": "encoders.3.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.weight": "encoders.3.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.bias": "encoders.3.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.weight": "encoders.3.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.bias": "encoders.3.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.weight": "encoders.3.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.bias": "encoders.3.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight": "encoders.3.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.bias": "encoders.3.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.weight": "encoders.3.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.bias": "encoders.4.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.weight": "encoders.4.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.bias": "encoders.4.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.weight": "encoders.4.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.bias": "encoders.4.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.weight": "encoders.4.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.bias": "encoders.4.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.weight": "encoders.4.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.bias": "encoders.4.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.weight": "encoders.4.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.bias": "encoders.4.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.weight": "encoders.4.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.bias": "encoders.4.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.weight": "encoders.4.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.bias": "encoders.4.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.weight": "encoders.4.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.bias": "encoders.5.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.weight": "encoders.5.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.bias": "encoders.5.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.weight": "encoders.5.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.bias": "encoders.5.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.weight": "encoders.5.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.bias": "encoders.5.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.weight": "encoders.5.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.bias": "encoders.5.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight": "encoders.5.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.bias": "encoders.5.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.weight": "encoders.5.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.bias": "encoders.5.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight": "encoders.5.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.bias": "encoders.5.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.weight": "encoders.5.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.bias": "encoders.6.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.weight": "encoders.6.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.bias": "encoders.6.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.weight": "encoders.6.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.bias": "encoders.6.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.weight": "encoders.6.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.bias": "encoders.6.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.weight": "encoders.6.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.bias": "encoders.6.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.weight": "encoders.6.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.bias": "encoders.6.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.weight": "encoders.6.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.bias": "encoders.6.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.weight": "encoders.6.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.bias": "encoders.6.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.weight": "encoders.6.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.bias": "encoders.7.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.weight": "encoders.7.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.bias": "encoders.7.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.weight": "encoders.7.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.bias": "encoders.7.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.weight": "encoders.7.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.bias": "encoders.7.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.weight": "encoders.7.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.bias": "encoders.7.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.weight": "encoders.7.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.bias": "encoders.7.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.weight": "encoders.7.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.bias": "encoders.7.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.weight": "encoders.7.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.bias": "encoders.7.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.weight": "encoders.7.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.bias": "encoders.8.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.weight": "encoders.8.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.bias": "encoders.8.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.weight": "encoders.8.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.bias": "encoders.8.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.weight": "encoders.8.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.bias": "encoders.8.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.weight": "encoders.8.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.bias": "encoders.8.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.weight": "encoders.8.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.bias": "encoders.8.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.weight": "encoders.8.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.bias": "encoders.8.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.weight": "encoders.8.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.bias": "encoders.8.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.weight": "encoders.8.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.bias": "encoders.9.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.weight": "encoders.9.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.bias": "encoders.9.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.weight": "encoders.9.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.bias": "encoders.9.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.weight": "encoders.9.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.bias": "encoders.9.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.weight": "encoders.9.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.bias": "encoders.9.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.weight": "encoders.9.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.bias": "encoders.9.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.weight": "encoders.9.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.bias": "encoders.9.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.weight": "encoders.9.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.bias": "encoders.9.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight": "encoders.9.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.final_layer_norm.bias": "final_layer_norm.bias",
            "cond_stage_model.transformer.text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": "position_embeds"
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
        return state_dict_
