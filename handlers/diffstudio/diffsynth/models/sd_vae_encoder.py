import torch
from .sd_unet import ResnetBlock, DownSampler
from .sd_vae_decoder import VAEAttentionBlock
from .tiler import TileWorker


class SDVAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.18215
        self.quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1)
        self.conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # DownEncoderBlock2D
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            DownSampler(128, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(128, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            DownSampler(256, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(256, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            DownSampler(512, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(512, 8, kernel_size=3, padding=1)

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
        hidden_states = self.conv_in(sample)
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
        hidden_states = self.quant_conv(hidden_states)
        hidden_states = hidden_states[:, :4]
        hidden_states *= self.scaling_factor

        return hidden_states
    
    def state_dict_converter(self):
        return SDVAEEncoderStateDictConverter()
    

class SDVAEEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # architecture
        block_types = [
            'ResnetBlock', 'ResnetBlock', 'DownSampler',
            'ResnetBlock', 'ResnetBlock', 'DownSampler',
            'ResnetBlock', 'ResnetBlock', 'DownSampler',
            'ResnetBlock', 'ResnetBlock',
            'ResnetBlock', 'VAEAttentionBlock', 'ResnetBlock'
        ]

        # Rename each parameter
        local_rename_dict = {
            "quant_conv": "quant_conv",
            "encoder.conv_in": "conv_in",
            "encoder.mid_block.attentions.0.group_norm": "blocks.12.norm",
            "encoder.mid_block.attentions.0.to_q": "blocks.12.transformer_blocks.0.to_q",
            "encoder.mid_block.attentions.0.to_k": "blocks.12.transformer_blocks.0.to_k",
            "encoder.mid_block.attentions.0.to_v": "blocks.12.transformer_blocks.0.to_v",
            "encoder.mid_block.attentions.0.to_out.0": "blocks.12.transformer_blocks.0.to_out",
            "encoder.mid_block.resnets.0.norm1": "blocks.11.norm1",
            "encoder.mid_block.resnets.0.conv1": "blocks.11.conv1",
            "encoder.mid_block.resnets.0.norm2": "blocks.11.norm2",
            "encoder.mid_block.resnets.0.conv2": "blocks.11.conv2",
            "encoder.mid_block.resnets.1.norm1": "blocks.13.norm1",
            "encoder.mid_block.resnets.1.conv1": "blocks.13.conv1",
            "encoder.mid_block.resnets.1.norm2": "blocks.13.norm2",
            "encoder.mid_block.resnets.1.conv2": "blocks.13.conv2",
            "encoder.conv_norm_out": "conv_norm_out",
            "encoder.conv_out": "conv_out",
        }
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            name_prefix = ".".join(names[:-1])
            if name_prefix in local_rename_dict:
                rename_dict[name] = local_rename_dict[name_prefix] + "." + names[-1]
            elif name.startswith("encoder.down_blocks"):
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
            "first_stage_model.encoder.conv_in.bias": "conv_in.bias",
            "first_stage_model.encoder.conv_in.weight": "conv_in.weight",
            "first_stage_model.encoder.conv_out.bias": "conv_out.bias",
            "first_stage_model.encoder.conv_out.weight": "conv_out.weight",
            "first_stage_model.encoder.down.0.block.0.conv1.bias": "blocks.0.conv1.bias",
            "first_stage_model.encoder.down.0.block.0.conv1.weight": "blocks.0.conv1.weight",
            "first_stage_model.encoder.down.0.block.0.conv2.bias": "blocks.0.conv2.bias",
            "first_stage_model.encoder.down.0.block.0.conv2.weight": "blocks.0.conv2.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.bias": "blocks.0.norm1.bias",
            "first_stage_model.encoder.down.0.block.0.norm1.weight": "blocks.0.norm1.weight",
            "first_stage_model.encoder.down.0.block.0.norm2.bias": "blocks.0.norm2.bias",
            "first_stage_model.encoder.down.0.block.0.norm2.weight": "blocks.0.norm2.weight",
            "first_stage_model.encoder.down.0.block.1.conv1.bias": "blocks.1.conv1.bias",
            "first_stage_model.encoder.down.0.block.1.conv1.weight": "blocks.1.conv1.weight",
            "first_stage_model.encoder.down.0.block.1.conv2.bias": "blocks.1.conv2.bias",
            "first_stage_model.encoder.down.0.block.1.conv2.weight": "blocks.1.conv2.weight",
            "first_stage_model.encoder.down.0.block.1.norm1.bias": "blocks.1.norm1.bias",
            "first_stage_model.encoder.down.0.block.1.norm1.weight": "blocks.1.norm1.weight",
            "first_stage_model.encoder.down.0.block.1.norm2.bias": "blocks.1.norm2.bias",
            "first_stage_model.encoder.down.0.block.1.norm2.weight": "blocks.1.norm2.weight",
            "first_stage_model.encoder.down.0.downsample.conv.bias": "blocks.2.conv.bias",
            "first_stage_model.encoder.down.0.downsample.conv.weight": "blocks.2.conv.weight",
            "first_stage_model.encoder.down.1.block.0.conv1.bias": "blocks.3.conv1.bias",
            "first_stage_model.encoder.down.1.block.0.conv1.weight": "blocks.3.conv1.weight",
            "first_stage_model.encoder.down.1.block.0.conv2.bias": "blocks.3.conv2.bias",
            "first_stage_model.encoder.down.1.block.0.conv2.weight": "blocks.3.conv2.weight",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.bias": "blocks.3.conv_shortcut.bias",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.weight": "blocks.3.conv_shortcut.weight",
            "first_stage_model.encoder.down.1.block.0.norm1.bias": "blocks.3.norm1.bias",
            "first_stage_model.encoder.down.1.block.0.norm1.weight": "blocks.3.norm1.weight",
            "first_stage_model.encoder.down.1.block.0.norm2.bias": "blocks.3.norm2.bias",
            "first_stage_model.encoder.down.1.block.0.norm2.weight": "blocks.3.norm2.weight",
            "first_stage_model.encoder.down.1.block.1.conv1.bias": "blocks.4.conv1.bias",
            "first_stage_model.encoder.down.1.block.1.conv1.weight": "blocks.4.conv1.weight",
            "first_stage_model.encoder.down.1.block.1.conv2.bias": "blocks.4.conv2.bias",
            "first_stage_model.encoder.down.1.block.1.conv2.weight": "blocks.4.conv2.weight",
            "first_stage_model.encoder.down.1.block.1.norm1.bias": "blocks.4.norm1.bias",
            "first_stage_model.encoder.down.1.block.1.norm1.weight": "blocks.4.norm1.weight",
            "first_stage_model.encoder.down.1.block.1.norm2.bias": "blocks.4.norm2.bias",
            "first_stage_model.encoder.down.1.block.1.norm2.weight": "blocks.4.norm2.weight",
            "first_stage_model.encoder.down.1.downsample.conv.bias": "blocks.5.conv.bias",
            "first_stage_model.encoder.down.1.downsample.conv.weight": "blocks.5.conv.weight",
            "first_stage_model.encoder.down.2.block.0.conv1.bias": "blocks.6.conv1.bias",
            "first_stage_model.encoder.down.2.block.0.conv1.weight": "blocks.6.conv1.weight",
            "first_stage_model.encoder.down.2.block.0.conv2.bias": "blocks.6.conv2.bias",
            "first_stage_model.encoder.down.2.block.0.conv2.weight": "blocks.6.conv2.weight",
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.bias": "blocks.6.conv_shortcut.bias",
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.weight": "blocks.6.conv_shortcut.weight",
            "first_stage_model.encoder.down.2.block.0.norm1.bias": "blocks.6.norm1.bias",
            "first_stage_model.encoder.down.2.block.0.norm1.weight": "blocks.6.norm1.weight",
            "first_stage_model.encoder.down.2.block.0.norm2.bias": "blocks.6.norm2.bias",
            "first_stage_model.encoder.down.2.block.0.norm2.weight": "blocks.6.norm2.weight",
            "first_stage_model.encoder.down.2.block.1.conv1.bias": "blocks.7.conv1.bias",
            "first_stage_model.encoder.down.2.block.1.conv1.weight": "blocks.7.conv1.weight",
            "first_stage_model.encoder.down.2.block.1.conv2.bias": "blocks.7.conv2.bias",
            "first_stage_model.encoder.down.2.block.1.conv2.weight": "blocks.7.conv2.weight",
            "first_stage_model.encoder.down.2.block.1.norm1.bias": "blocks.7.norm1.bias",
            "first_stage_model.encoder.down.2.block.1.norm1.weight": "blocks.7.norm1.weight",
            "first_stage_model.encoder.down.2.block.1.norm2.bias": "blocks.7.norm2.bias",
            "first_stage_model.encoder.down.2.block.1.norm2.weight": "blocks.7.norm2.weight",
            "first_stage_model.encoder.down.2.downsample.conv.bias": "blocks.8.conv.bias",
            "first_stage_model.encoder.down.2.downsample.conv.weight": "blocks.8.conv.weight",
            "first_stage_model.encoder.down.3.block.0.conv1.bias": "blocks.9.conv1.bias",
            "first_stage_model.encoder.down.3.block.0.conv1.weight": "blocks.9.conv1.weight",
            "first_stage_model.encoder.down.3.block.0.conv2.bias": "blocks.9.conv2.bias",
            "first_stage_model.encoder.down.3.block.0.conv2.weight": "blocks.9.conv2.weight",
            "first_stage_model.encoder.down.3.block.0.norm1.bias": "blocks.9.norm1.bias",
            "first_stage_model.encoder.down.3.block.0.norm1.weight": "blocks.9.norm1.weight",
            "first_stage_model.encoder.down.3.block.0.norm2.bias": "blocks.9.norm2.bias",
            "first_stage_model.encoder.down.3.block.0.norm2.weight": "blocks.9.norm2.weight",
            "first_stage_model.encoder.down.3.block.1.conv1.bias": "blocks.10.conv1.bias",
            "first_stage_model.encoder.down.3.block.1.conv1.weight": "blocks.10.conv1.weight",
            "first_stage_model.encoder.down.3.block.1.conv2.bias": "blocks.10.conv2.bias",
            "first_stage_model.encoder.down.3.block.1.conv2.weight": "blocks.10.conv2.weight",
            "first_stage_model.encoder.down.3.block.1.norm1.bias": "blocks.10.norm1.bias",
            "first_stage_model.encoder.down.3.block.1.norm1.weight": "blocks.10.norm1.weight",
            "first_stage_model.encoder.down.3.block.1.norm2.bias": "blocks.10.norm2.bias",
            "first_stage_model.encoder.down.3.block.1.norm2.weight": "blocks.10.norm2.weight",
            "first_stage_model.encoder.mid.attn_1.k.bias": "blocks.12.transformer_blocks.0.to_k.bias",
            "first_stage_model.encoder.mid.attn_1.k.weight": "blocks.12.transformer_blocks.0.to_k.weight",
            "first_stage_model.encoder.mid.attn_1.norm.bias": "blocks.12.norm.bias",
            "first_stage_model.encoder.mid.attn_1.norm.weight": "blocks.12.norm.weight",
            "first_stage_model.encoder.mid.attn_1.proj_out.bias": "blocks.12.transformer_blocks.0.to_out.bias",       
            "first_stage_model.encoder.mid.attn_1.proj_out.weight": "blocks.12.transformer_blocks.0.to_out.weight",   
            "first_stage_model.encoder.mid.attn_1.q.bias": "blocks.12.transformer_blocks.0.to_q.bias",
            "first_stage_model.encoder.mid.attn_1.q.weight": "blocks.12.transformer_blocks.0.to_q.weight",
            "first_stage_model.encoder.mid.attn_1.v.bias": "blocks.12.transformer_blocks.0.to_v.bias",
            "first_stage_model.encoder.mid.attn_1.v.weight": "blocks.12.transformer_blocks.0.to_v.weight",
            "first_stage_model.encoder.mid.block_1.conv1.bias": "blocks.11.conv1.bias",
            "first_stage_model.encoder.mid.block_1.conv1.weight": "blocks.11.conv1.weight",
            "first_stage_model.encoder.mid.block_1.conv2.bias": "blocks.11.conv2.bias",
            "first_stage_model.encoder.mid.block_1.conv2.weight": "blocks.11.conv2.weight",
            "first_stage_model.encoder.mid.block_1.norm1.bias": "blocks.11.norm1.bias",
            "first_stage_model.encoder.mid.block_1.norm1.weight": "blocks.11.norm1.weight",
            "first_stage_model.encoder.mid.block_1.norm2.bias": "blocks.11.norm2.bias",
            "first_stage_model.encoder.mid.block_1.norm2.weight": "blocks.11.norm2.weight",
            "first_stage_model.encoder.mid.block_2.conv1.bias": "blocks.13.conv1.bias",
            "first_stage_model.encoder.mid.block_2.conv1.weight": "blocks.13.conv1.weight",
            "first_stage_model.encoder.mid.block_2.conv2.bias": "blocks.13.conv2.bias",
            "first_stage_model.encoder.mid.block_2.conv2.weight": "blocks.13.conv2.weight",
            "first_stage_model.encoder.mid.block_2.norm1.bias": "blocks.13.norm1.bias",
            "first_stage_model.encoder.mid.block_2.norm1.weight": "blocks.13.norm1.weight",
            "first_stage_model.encoder.mid.block_2.norm2.bias": "blocks.13.norm2.bias",
            "first_stage_model.encoder.mid.block_2.norm2.weight": "blocks.13.norm2.weight",
            "first_stage_model.encoder.norm_out.bias": "conv_norm_out.bias",
            "first_stage_model.encoder.norm_out.weight": "conv_norm_out.weight",
            "first_stage_model.quant_conv.bias": "quant_conv.bias",
            "first_stage_model.quant_conv.weight": "quant_conv.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if "transformer_blocks" in rename_dict[name]:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_
