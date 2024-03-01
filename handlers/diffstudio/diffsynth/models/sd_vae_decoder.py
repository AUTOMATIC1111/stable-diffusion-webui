import torch
from .attention import Attention
from .sd_unet import ResnetBlock, UpSampler
from .tiler import TileWorker


class VAEAttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        self.transformer_blocks = torch.nn.ModuleList([
            Attention(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                bias_q=True,
                bias_kv=True,
                bias_out=True
            )
            for d in range(num_layers)
        ])

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class SDVAEDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.18215
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.conv_in = torch.nn.Conv2d(4, 512, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D
            ResnetBlock(512, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            UpSampler(256),
            # UpDecoderBlock2D
            ResnetBlock(256, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-5)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
    
    def tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. pre-process
        sample = sample / self.scaling_factor
        hidden_states = self.post_quant_conv(sample)
        hidden_states = self.conv_in(hidden_states)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
        
        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states
    
    def state_dict_converter(self):
        return SDVAEDecoderStateDictConverter()
    

class SDVAEDecoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # architecture
        block_types = [
            'ResnetBlock', 'VAEAttentionBlock', 'ResnetBlock',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock', 'UpSampler',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock', 'UpSampler',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock', 'UpSampler',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock'
        ]

        # Rename each parameter
        local_rename_dict = {
            "post_quant_conv": "post_quant_conv",
            "decoder.conv_in": "conv_in",
            "decoder.mid_block.attentions.0.group_norm": "blocks.1.norm",
            "decoder.mid_block.attentions.0.to_q": "blocks.1.transformer_blocks.0.to_q",
            "decoder.mid_block.attentions.0.to_k": "blocks.1.transformer_blocks.0.to_k",
            "decoder.mid_block.attentions.0.to_v": "blocks.1.transformer_blocks.0.to_v",
            "decoder.mid_block.attentions.0.to_out.0": "blocks.1.transformer_blocks.0.to_out",
            "decoder.mid_block.resnets.0.norm1": "blocks.0.norm1",
            "decoder.mid_block.resnets.0.conv1": "blocks.0.conv1",
            "decoder.mid_block.resnets.0.norm2": "blocks.0.norm2",
            "decoder.mid_block.resnets.0.conv2": "blocks.0.conv2",
            "decoder.mid_block.resnets.1.norm1": "blocks.2.norm1",
            "decoder.mid_block.resnets.1.conv1": "blocks.2.conv1",
            "decoder.mid_block.resnets.1.norm2": "blocks.2.norm2",
            "decoder.mid_block.resnets.1.conv2": "blocks.2.conv2",
            "decoder.conv_norm_out": "conv_norm_out",
            "decoder.conv_out": "conv_out",
        }
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": 2, "DownSampler": 2, "UpSampler": 2}
        last_block_type_with_id = {"ResnetBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            name_prefix = ".".join(names[:-1])
            if name_prefix in local_rename_dict:
                rename_dict[name] = local_rename_dict[name_prefix] + "." + names[-1]
            elif name.startswith("decoder.up_blocks"):
                block_type = {"resnets": "ResnetBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[3]]
                block_type_with_id = ".".join(names[:5])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:5])
                names = ["blocks", str(block_id[block_type])] + names[5:]
                rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "first_stage_model.decoder.conv_in.bias": "conv_in.bias",
            "first_stage_model.decoder.conv_in.weight": "conv_in.weight",
            "first_stage_model.decoder.conv_out.bias": "conv_out.bias",
            "first_stage_model.decoder.conv_out.weight": "conv_out.weight",
            "first_stage_model.decoder.mid.attn_1.k.bias": "blocks.1.transformer_blocks.0.to_k.bias",
            "first_stage_model.decoder.mid.attn_1.k.weight": "blocks.1.transformer_blocks.0.to_k.weight",
            "first_stage_model.decoder.mid.attn_1.norm.bias": "blocks.1.norm.bias",
            "first_stage_model.decoder.mid.attn_1.norm.weight": "blocks.1.norm.weight",
            "first_stage_model.decoder.mid.attn_1.proj_out.bias": "blocks.1.transformer_blocks.0.to_out.bias",    
            "first_stage_model.decoder.mid.attn_1.proj_out.weight": "blocks.1.transformer_blocks.0.to_out.weight",
            "first_stage_model.decoder.mid.attn_1.q.bias": "blocks.1.transformer_blocks.0.to_q.bias",
            "first_stage_model.decoder.mid.attn_1.q.weight": "blocks.1.transformer_blocks.0.to_q.weight",
            "first_stage_model.decoder.mid.attn_1.v.bias": "blocks.1.transformer_blocks.0.to_v.bias",
            "first_stage_model.decoder.mid.attn_1.v.weight": "blocks.1.transformer_blocks.0.to_v.weight",
            "first_stage_model.decoder.mid.block_1.conv1.bias": "blocks.0.conv1.bias",
            "first_stage_model.decoder.mid.block_1.conv1.weight": "blocks.0.conv1.weight",
            "first_stage_model.decoder.mid.block_1.conv2.bias": "blocks.0.conv2.bias",
            "first_stage_model.decoder.mid.block_1.conv2.weight": "blocks.0.conv2.weight",
            "first_stage_model.decoder.mid.block_1.norm1.bias": "blocks.0.norm1.bias",
            "first_stage_model.decoder.mid.block_1.norm1.weight": "blocks.0.norm1.weight",
            "first_stage_model.decoder.mid.block_1.norm2.bias": "blocks.0.norm2.bias",
            "first_stage_model.decoder.mid.block_1.norm2.weight": "blocks.0.norm2.weight",
            "first_stage_model.decoder.mid.block_2.conv1.bias": "blocks.2.conv1.bias",
            "first_stage_model.decoder.mid.block_2.conv1.weight": "blocks.2.conv1.weight",
            "first_stage_model.decoder.mid.block_2.conv2.bias": "blocks.2.conv2.bias",
            "first_stage_model.decoder.mid.block_2.conv2.weight": "blocks.2.conv2.weight",
            "first_stage_model.decoder.mid.block_2.norm1.bias": "blocks.2.norm1.bias",
            "first_stage_model.decoder.mid.block_2.norm1.weight": "blocks.2.norm1.weight",
            "first_stage_model.decoder.mid.block_2.norm2.bias": "blocks.2.norm2.bias",
            "first_stage_model.decoder.mid.block_2.norm2.weight": "blocks.2.norm2.weight",
            "first_stage_model.decoder.norm_out.bias": "conv_norm_out.bias",
            "first_stage_model.decoder.norm_out.weight": "conv_norm_out.weight",
            "first_stage_model.decoder.up.0.block.0.conv1.bias": "blocks.15.conv1.bias",
            "first_stage_model.decoder.up.0.block.0.conv1.weight": "blocks.15.conv1.weight",
            "first_stage_model.decoder.up.0.block.0.conv2.bias": "blocks.15.conv2.bias",
            "first_stage_model.decoder.up.0.block.0.conv2.weight": "blocks.15.conv2.weight",
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.bias": "blocks.15.conv_shortcut.bias",
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.weight": "blocks.15.conv_shortcut.weight",       
            "first_stage_model.decoder.up.0.block.0.norm1.bias": "blocks.15.norm1.bias",
            "first_stage_model.decoder.up.0.block.0.norm1.weight": "blocks.15.norm1.weight",
            "first_stage_model.decoder.up.0.block.0.norm2.bias": "blocks.15.norm2.bias",
            "first_stage_model.decoder.up.0.block.0.norm2.weight": "blocks.15.norm2.weight",
            "first_stage_model.decoder.up.0.block.1.conv1.bias": "blocks.16.conv1.bias",
            "first_stage_model.decoder.up.0.block.1.conv1.weight": "blocks.16.conv1.weight",
            "first_stage_model.decoder.up.0.block.1.conv2.bias": "blocks.16.conv2.bias",
            "first_stage_model.decoder.up.0.block.1.conv2.weight": "blocks.16.conv2.weight",
            "first_stage_model.decoder.up.0.block.1.norm1.bias": "blocks.16.norm1.bias",
            "first_stage_model.decoder.up.0.block.1.norm1.weight": "blocks.16.norm1.weight",
            "first_stage_model.decoder.up.0.block.1.norm2.bias": "blocks.16.norm2.bias",
            "first_stage_model.decoder.up.0.block.1.norm2.weight": "blocks.16.norm2.weight",
            "first_stage_model.decoder.up.0.block.2.conv1.bias": "blocks.17.conv1.bias",
            "first_stage_model.decoder.up.0.block.2.conv1.weight": "blocks.17.conv1.weight",
            "first_stage_model.decoder.up.0.block.2.conv2.bias": "blocks.17.conv2.bias",
            "first_stage_model.decoder.up.0.block.2.conv2.weight": "blocks.17.conv2.weight",
            "first_stage_model.decoder.up.0.block.2.norm1.bias": "blocks.17.norm1.bias",
            "first_stage_model.decoder.up.0.block.2.norm1.weight": "blocks.17.norm1.weight",
            "first_stage_model.decoder.up.0.block.2.norm2.bias": "blocks.17.norm2.bias",
            "first_stage_model.decoder.up.0.block.2.norm2.weight": "blocks.17.norm2.weight",
            "first_stage_model.decoder.up.1.block.0.conv1.bias": "blocks.11.conv1.bias",
            "first_stage_model.decoder.up.1.block.0.conv1.weight": "blocks.11.conv1.weight",
            "first_stage_model.decoder.up.1.block.0.conv2.bias": "blocks.11.conv2.bias",
            "first_stage_model.decoder.up.1.block.0.conv2.weight": "blocks.11.conv2.weight",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.bias": "blocks.11.conv_shortcut.bias",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.weight": "blocks.11.conv_shortcut.weight",       
            "first_stage_model.decoder.up.1.block.0.norm1.bias": "blocks.11.norm1.bias",
            "first_stage_model.decoder.up.1.block.0.norm1.weight": "blocks.11.norm1.weight",
            "first_stage_model.decoder.up.1.block.0.norm2.bias": "blocks.11.norm2.bias",
            "first_stage_model.decoder.up.1.block.0.norm2.weight": "blocks.11.norm2.weight",
            "first_stage_model.decoder.up.1.block.1.conv1.bias": "blocks.12.conv1.bias",
            "first_stage_model.decoder.up.1.block.1.conv1.weight": "blocks.12.conv1.weight",
            "first_stage_model.decoder.up.1.block.1.conv2.bias": "blocks.12.conv2.bias",
            "first_stage_model.decoder.up.1.block.1.conv2.weight": "blocks.12.conv2.weight",
            "first_stage_model.decoder.up.1.block.1.norm1.bias": "blocks.12.norm1.bias",
            "first_stage_model.decoder.up.1.block.1.norm1.weight": "blocks.12.norm1.weight",
            "first_stage_model.decoder.up.1.block.1.norm2.bias": "blocks.12.norm2.bias",
            "first_stage_model.decoder.up.1.block.1.norm2.weight": "blocks.12.norm2.weight",
            "first_stage_model.decoder.up.1.block.2.conv1.bias": "blocks.13.conv1.bias",
            "first_stage_model.decoder.up.1.block.2.conv1.weight": "blocks.13.conv1.weight",
            "first_stage_model.decoder.up.1.block.2.conv2.bias": "blocks.13.conv2.bias",
            "first_stage_model.decoder.up.1.block.2.conv2.weight": "blocks.13.conv2.weight",
            "first_stage_model.decoder.up.1.block.2.norm1.bias": "blocks.13.norm1.bias",
            "first_stage_model.decoder.up.1.block.2.norm1.weight": "blocks.13.norm1.weight",
            "first_stage_model.decoder.up.1.block.2.norm2.bias": "blocks.13.norm2.bias",
            "first_stage_model.decoder.up.1.block.2.norm2.weight": "blocks.13.norm2.weight",
            "first_stage_model.decoder.up.1.upsample.conv.bias": "blocks.14.conv.bias",
            "first_stage_model.decoder.up.1.upsample.conv.weight": "blocks.14.conv.weight",
            "first_stage_model.decoder.up.2.block.0.conv1.bias": "blocks.7.conv1.bias",
            "first_stage_model.decoder.up.2.block.0.conv1.weight": "blocks.7.conv1.weight",
            "first_stage_model.decoder.up.2.block.0.conv2.bias": "blocks.7.conv2.bias",
            "first_stage_model.decoder.up.2.block.0.conv2.weight": "blocks.7.conv2.weight",
            "first_stage_model.decoder.up.2.block.0.norm1.bias": "blocks.7.norm1.bias",
            "first_stage_model.decoder.up.2.block.0.norm1.weight": "blocks.7.norm1.weight",
            "first_stage_model.decoder.up.2.block.0.norm2.bias": "blocks.7.norm2.bias",
            "first_stage_model.decoder.up.2.block.0.norm2.weight": "blocks.7.norm2.weight",
            "first_stage_model.decoder.up.2.block.1.conv1.bias": "blocks.8.conv1.bias",
            "first_stage_model.decoder.up.2.block.1.conv1.weight": "blocks.8.conv1.weight",
            "first_stage_model.decoder.up.2.block.1.conv2.bias": "blocks.8.conv2.bias",
            "first_stage_model.decoder.up.2.block.1.conv2.weight": "blocks.8.conv2.weight",
            "first_stage_model.decoder.up.2.block.1.norm1.bias": "blocks.8.norm1.bias",
            "first_stage_model.decoder.up.2.block.1.norm1.weight": "blocks.8.norm1.weight",
            "first_stage_model.decoder.up.2.block.1.norm2.bias": "blocks.8.norm2.bias",
            "first_stage_model.decoder.up.2.block.1.norm2.weight": "blocks.8.norm2.weight",
            "first_stage_model.decoder.up.2.block.2.conv1.bias": "blocks.9.conv1.bias",
            "first_stage_model.decoder.up.2.block.2.conv1.weight": "blocks.9.conv1.weight",
            "first_stage_model.decoder.up.2.block.2.conv2.bias": "blocks.9.conv2.bias",
            "first_stage_model.decoder.up.2.block.2.conv2.weight": "blocks.9.conv2.weight",
            "first_stage_model.decoder.up.2.block.2.norm1.bias": "blocks.9.norm1.bias",
            "first_stage_model.decoder.up.2.block.2.norm1.weight": "blocks.9.norm1.weight",
            "first_stage_model.decoder.up.2.block.2.norm2.bias": "blocks.9.norm2.bias",
            "first_stage_model.decoder.up.2.block.2.norm2.weight": "blocks.9.norm2.weight",
            "first_stage_model.decoder.up.2.upsample.conv.bias": "blocks.10.conv.bias",
            "first_stage_model.decoder.up.2.upsample.conv.weight": "blocks.10.conv.weight",
            "first_stage_model.decoder.up.3.block.0.conv1.bias": "blocks.3.conv1.bias",
            "first_stage_model.decoder.up.3.block.0.conv1.weight": "blocks.3.conv1.weight",
            "first_stage_model.decoder.up.3.block.0.conv2.bias": "blocks.3.conv2.bias",
            "first_stage_model.decoder.up.3.block.0.conv2.weight": "blocks.3.conv2.weight",
            "first_stage_model.decoder.up.3.block.0.norm1.bias": "blocks.3.norm1.bias",
            "first_stage_model.decoder.up.3.block.0.norm1.weight": "blocks.3.norm1.weight",
            "first_stage_model.decoder.up.3.block.0.norm2.bias": "blocks.3.norm2.bias",
            "first_stage_model.decoder.up.3.block.0.norm2.weight": "blocks.3.norm2.weight",
            "first_stage_model.decoder.up.3.block.1.conv1.bias": "blocks.4.conv1.bias",
            "first_stage_model.decoder.up.3.block.1.conv1.weight": "blocks.4.conv1.weight",
            "first_stage_model.decoder.up.3.block.1.conv2.bias": "blocks.4.conv2.bias",
            "first_stage_model.decoder.up.3.block.1.conv2.weight": "blocks.4.conv2.weight",
            "first_stage_model.decoder.up.3.block.1.norm1.bias": "blocks.4.norm1.bias",
            "first_stage_model.decoder.up.3.block.1.norm1.weight": "blocks.4.norm1.weight",
            "first_stage_model.decoder.up.3.block.1.norm2.bias": "blocks.4.norm2.bias",
            "first_stage_model.decoder.up.3.block.1.norm2.weight": "blocks.4.norm2.weight",
            "first_stage_model.decoder.up.3.block.2.conv1.bias": "blocks.5.conv1.bias",
            "first_stage_model.decoder.up.3.block.2.conv1.weight": "blocks.5.conv1.weight",
            "first_stage_model.decoder.up.3.block.2.conv2.bias": "blocks.5.conv2.bias",
            "first_stage_model.decoder.up.3.block.2.conv2.weight": "blocks.5.conv2.weight",
            "first_stage_model.decoder.up.3.block.2.norm1.bias": "blocks.5.norm1.bias",
            "first_stage_model.decoder.up.3.block.2.norm1.weight": "blocks.5.norm1.weight",
            "first_stage_model.decoder.up.3.block.2.norm2.bias": "blocks.5.norm2.bias",
            "first_stage_model.decoder.up.3.block.2.norm2.weight": "blocks.5.norm2.weight",
            "first_stage_model.decoder.up.3.upsample.conv.bias": "blocks.6.conv.bias",
            "first_stage_model.decoder.up.3.upsample.conv.weight": "blocks.6.conv.weight",
            "first_stage_model.post_quant_conv.bias": "post_quant_conv.bias",
            "first_stage_model.post_quant_conv.weight": "post_quant_conv.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if "transformer_blocks" in rename_dict[name]:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_
