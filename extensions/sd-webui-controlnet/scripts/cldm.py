import torch
import torch.nn as nn

from modules import devices


try:
    from sgm.modules.diffusionmodules.openaimodel import conv_nd, linear, zero_module, timestep_embedding, \
        TimestepEmbedSequential, ResBlock, Downsample, SpatialTransformer, exists
    using_sgm = True
except:
    from ldm.modules.diffusionmodules.openaimodel import conv_nd, linear, zero_module, timestep_embedding, \
        TimestepEmbedSequential, ResBlock, Downsample, SpatialTransformer, exists
    using_sgm = False


class PlugableControlModel(nn.Module):
    def __init__(self, config, state_dict=None):
        super().__init__()
        self.config = config
        self.control_model = ControlNet(**self.config).cpu()
        if state_dict is not None:
            self.control_model.load_state_dict(state_dict, strict=False)
        self.gpu_component = None
        self.is_control_lora = False

    def reset(self):
        pass
            
    def forward(self, *args, **kwargs):
        return self.control_model(*args, **kwargs)

    def aggressive_lowvram(self):
        self.to('cpu')

        def send_me_to_gpu(module, _):
            if self.gpu_component == module:
                return

            if self.gpu_component is not None:
                self.gpu_component.to('cpu')

            module.to(devices.get_device_for("controlnet"))
            self.gpu_component = module

        self.control_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        self.control_model.input_hint_block.register_forward_pre_hook(send_me_to_gpu)
        self.control_model.label_emb.register_forward_pre_hook(send_me_to_gpu)
        for m in self.control_model.input_blocks:
            m.register_forward_pre_hook(send_me_to_gpu)
        for m in self.control_model.zero_convs:
            m.register_forward_pre_hook(send_me_to_gpu)
        self.control_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        self.control_model.middle_block_out.register_forward_pre_hook(send_me_to_gpu)
        return

    def fullvram(self):
        self.to(devices.get_device_for("controlnet"))
        return
            

class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=False,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        device=None,
        global_average_pooling=False,
    ):
        super().__init__()

        self.global_average_pooling = global_average_pooling

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle = transformer_depth[-1]
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
                    conv_nd(dims, hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        original_type = x.dtype

        x = x.to(self.dtype)
        hint = hint.to(self.dtype)
        timesteps = timesteps.to(self.dtype)
        context = context.to(self.dtype)

        if y is not None:
            y = y.to(self.dtype)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)
        outs = []

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        outs = [o.to(original_type) for o in outs]

        return outs
