# cond_imageをU-Netのforwardで渡すバージョンのControlNet-LLLite検証用実装
# ControlNet-LLLite implementation for verification with cond_image passed in U-Net's forward

import os
import re
from typing import Optional, List, Type
import torch
from library import sdxl_original_unet


# input_blocksに適用するかどうか / if True, input_blocks are not applied
SKIP_INPUT_BLOCKS = False

# output_blocksに適用するかどうか / if True, output_blocks are not applied
SKIP_OUTPUT_BLOCKS = True

# conv2dに適用するかどうか / if True, conv2d are not applied
SKIP_CONV2D = False

# transformer_blocksのみに適用するかどうか。Trueの場合、ResBlockには適用されない
# if True, only transformer_blocks are applied, and ResBlocks are not applied
TRANSFORMER_ONLY = True  # if True, SKIP_CONV2D is ignored because conv2d is not used in transformer_blocks

# Trueならattn1とattn2にのみ適用し、ffなどには適用しない / if True, apply only to attn1 and attn2, not to ff etc.
ATTN1_2_ONLY = True

# Trueならattn1のQKV、attn2のQにのみ適用する、ATTN1_2_ONLY指定時のみ有効 / if True, apply only to attn1 QKV and attn2 Q, only valid when ATTN1_2_ONLY is specified
ATTN_QKV_ONLY = True

# Trueならattn1やffなどにのみ適用し、attn2などには適用しない / if True, apply only to attn1 and ff, not to attn2
# ATTN1_2_ONLYと同時にTrueにできない / cannot be True at the same time as ATTN1_2_ONLY
ATTN1_ETC_ONLY = False  # True

# transformer_blocksの最大インデックス。Noneなら全てのtransformer_blocksに適用
# max index of transformer_blocks. if None, apply to all transformer_blocks
TRANSFORMER_MAX_BLOCK_INDEX = None

ORIGINAL_LINEAR = torch.nn.Linear
ORIGINAL_CONV2D = torch.nn.Conv2d


def add_lllite_modules(module: torch.nn.Module, in_dim: int, depth, cond_emb_dim, mlp_dim) -> None:
    # conditioning1はconditioning imageを embedding する。timestepごとに呼ばれない
    # conditioning1 embeds conditioning image. it is not called for each timestep
    modules = []
    modules.append(ORIGINAL_CONV2D(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size
    if depth == 1:
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(ORIGINAL_CONV2D(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
    elif depth == 2:
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(ORIGINAL_CONV2D(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
    elif depth == 3:
        # kernel size 8は大きすぎるので、4にする / kernel size 8 is too large, so set it to 4
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(ORIGINAL_CONV2D(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(ORIGINAL_CONV2D(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

    module.lllite_conditioning1 = torch.nn.Sequential(*modules)

    # downで入力の次元数を削減する。LoRAにヒントを得ていることにする
    # midでconditioning image embeddingと入力を結合する
    # upで元の次元数に戻す
    # これらはtimestepごとに呼ばれる
    # reduce the number of input dimensions with down. inspired by LoRA
    # combine conditioning image embedding and input with mid
    # restore to the original dimension with up
    # these are called for each timestep

    module.lllite_down = torch.nn.Sequential(
        ORIGINAL_LINEAR(in_dim, mlp_dim),
        torch.nn.ReLU(inplace=True),
    )
    module.lllite_mid = torch.nn.Sequential(
        ORIGINAL_LINEAR(mlp_dim + cond_emb_dim, mlp_dim),
        torch.nn.ReLU(inplace=True),
    )
    module.lllite_up = torch.nn.Sequential(
        ORIGINAL_LINEAR(mlp_dim, in_dim),
    )

    # Zero-Convにする / set to Zero-Conv
    torch.nn.init.zeros_(module.lllite_up[0].weight)  # zero conv


class LLLiteLinear(ORIGINAL_LINEAR):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.enabled = False

    def set_lllite(self, depth, cond_emb_dim, name, mlp_dim, dropout=None, multiplier=1.0):
        self.enabled = True
        self.lllite_name = name
        self.cond_emb_dim = cond_emb_dim
        self.dropout = dropout
        self.multiplier = multiplier  # ignored

        in_dim = self.in_features
        add_lllite_modules(self, in_dim, depth, cond_emb_dim, mlp_dim)

        self.cond_image = None
        self.cond_emb = None

    def set_cond_image(self, cond_image):
        self.cond_image = cond_image
        self.cond_emb = None

    def forward(self, x):
        if not self.enabled:
            return super().forward(x)

        if self.cond_emb is None:
            self.cond_emb = self.lllite_conditioning1(self.cond_image)
        cx = self.cond_emb

        # reshape / b,c,h,w -> b,h*w,c
        n, c, h, w = cx.shape
        cx = cx.view(n, c, h * w).permute(0, 2, 1)

        cx = torch.cat([cx, self.lllite_down(x)], dim=2)
        cx = self.lllite_mid(cx)

        if self.dropout is not None and self.training:
            cx = torch.nn.functional.dropout(cx, p=self.dropout)

        cx = self.lllite_up(cx) * self.multiplier

        x = super().forward(x + cx)  # ここで元のモジュールを呼び出す / call the original module here
        return x


class LLLiteConv2d(ORIGINAL_CONV2D):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.enabled = False

    def set_lllite(self, depth, cond_emb_dim, name, mlp_dim, dropout=None, multiplier=1.0):
        self.enabled = True
        self.lllite_name = name
        self.cond_emb_dim = cond_emb_dim
        self.dropout = dropout
        self.multiplier = multiplier  # ignored

        in_dim = self.in_channels
        add_lllite_modules(self, in_dim, depth, cond_emb_dim, mlp_dim)

        self.cond_image = None
        self.cond_emb = None

    def set_cond_image(self, cond_image):
        self.cond_image = cond_image
        self.cond_emb = None

    def forward(self, x):  # , cond_image=None):
        if not self.enabled:
            return super().forward(x)

        if self.cond_emb is None:
            self.cond_emb = self.lllite_conditioning1(self.cond_image)
        cx = self.cond_emb

        cx = torch.cat([cx, self.down(x)], dim=1)
        cx = self.mid(cx)

        if self.dropout is not None and self.training:
            cx = torch.nn.functional.dropout(cx, p=self.dropout)

        cx = self.up(cx) * self.multiplier

        x = super().forward(x + cx)  # ここで元のモジュールを呼び出す / call the original module here
        return x


class SdxlUNet2DConditionModelControlNetLLLite(sdxl_original_unet.SdxlUNet2DConditionModel):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    LLLITE_PREFIX = "lllite_unet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_lllite(
        self,
        cond_emb_dim: int = 16,
        mlp_dim: int = 16,
        dropout: Optional[float] = None,
        varbose: Optional[bool] = False,
        multiplier: Optional[float] = 1.0,
    ) -> None:
        def apply_to_modules(
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[torch.nn.Module]:
            prefix = "lllite_unet"

            modules = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "LLLiteLinear"
                        is_conv2d = child_module.__class__.__name__ == "LLLiteConv2d"

                        if is_linear or (is_conv2d and not SKIP_CONV2D):
                            # block indexからdepthを計算: depthはconditioningのサイズやチャネルを計算するのに使う
                            # block index to depth: depth is using to calculate conditioning size and channels
                            block_name, index1, index2 = (name + "." + child_name).split(".")[:3]
                            index1 = int(index1)
                            if block_name == "input_blocks":
                                if SKIP_INPUT_BLOCKS:
                                    continue
                                depth = 1 if index1 <= 2 else (2 if index1 <= 5 else 3)
                            elif block_name == "middle_block":
                                depth = 3
                            elif block_name == "output_blocks":
                                if SKIP_OUTPUT_BLOCKS:
                                    continue
                                depth = 3 if index1 <= 2 else (2 if index1 <= 5 else 1)
                                if int(index2) >= 2:
                                    depth -= 1
                            else:
                                raise NotImplementedError()

                            lllite_name = prefix + "." + name + "." + child_name
                            lllite_name = lllite_name.replace(".", "_")

                            if TRANSFORMER_MAX_BLOCK_INDEX is not None:
                                p = lllite_name.find("transformer_blocks")
                                if p >= 0:
                                    tf_index = int(lllite_name[p:].split("_")[2])
                                    if tf_index > TRANSFORMER_MAX_BLOCK_INDEX:
                                        continue

                            #  time embは適用外とする
                            # attn2のconditioning (CLIPからの入力) はshapeが違うので適用できない
                            # time emb is not applied
                            # attn2 conditioning (input from CLIP) cannot be applied because the shape is different
                            if "emb_layers" in lllite_name or (
                                "attn2" in lllite_name and ("to_k" in lllite_name or "to_v" in lllite_name)
                            ):
                                continue

                            if ATTN1_2_ONLY:
                                if not ("attn1" in lllite_name or "attn2" in lllite_name):
                                    continue
                                if ATTN_QKV_ONLY:
                                    if "to_out" in lllite_name:
                                        continue

                            if ATTN1_ETC_ONLY:
                                if "proj_out" in lllite_name:
                                    pass
                                elif "attn1" in lllite_name and (
                                    "to_k" in lllite_name or "to_v" in lllite_name or "to_out" in lllite_name
                                ):
                                    pass
                                elif "ff_net_2" in lllite_name:
                                    pass
                                else:
                                    continue

                            child_module.set_lllite(depth, cond_emb_dim, lllite_name, mlp_dim, dropout, multiplier)
                            modules.append(child_module)

            return modules

        target_modules = SdxlUNet2DConditionModelControlNetLLLite.UNET_TARGET_REPLACE_MODULE
        if not TRANSFORMER_ONLY:
            target_modules = target_modules + SdxlUNet2DConditionModelControlNetLLLite.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        # create module instances
        self.lllite_modules = apply_to_modules(self, target_modules)
        print(f"enable ControlNet LLLite for U-Net: {len(self.lllite_modules)} modules.")

    # def prepare_optimizer_params(self):
    def prepare_params(self):
        train_params = []
        non_train_params = []
        for name, p in self.named_parameters():
            if "lllite" in name:
                train_params.append(p)
            else:
                non_train_params.append(p)
        print(f"count of trainable parameters: {len(train_params)}")
        print(f"count of non-trainable parameters: {len(non_train_params)}")

        for p in non_train_params:
            p.requires_grad_(False)

        # without this, an error occurs in the optimizer
        #       RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        non_train_params[0].requires_grad_(True)

        for p in train_params:
            p.requires_grad_(True)

        return train_params

    # def prepare_grad_etc(self):
    #     self.requires_grad_(True)

    # def on_epoch_start(self):
    #     self.train()

    def get_trainable_params(self):
        return [p[1] for p in self.named_parameters() if "lllite" in p[0]]

    def save_lllite_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        org_state_dict = self.state_dict()

        # copy LLLite keys from org_state_dict to state_dict with key conversion
        state_dict = {}
        for key in org_state_dict.keys():
            # split with ".lllite"
            pos = key.find(".lllite")
            if pos < 0:
                continue
            lllite_key = SdxlUNet2DConditionModelControlNetLLLite.LLLITE_PREFIX + "." + key[:pos]
            lllite_key = lllite_key.replace(".", "_") + key[pos:]
            lllite_key = lllite_key.replace(".lllite_", ".")
            state_dict[lllite_key] = org_state_dict[key]

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_lllite_weights(self, file, non_lllite_unet_sd=None):
        r"""
        LLLiteの重みを読み込まない（initされた値を使う）場合はfileにNoneを指定する。
        この場合、non_lllite_unet_sdにはU-Netのstate_dictを指定する。

        If you do not want to load LLLite weights (use initialized values), specify None for file.
        In this case, specify the state_dict of U-Net for non_lllite_unet_sd.
        """
        if not file:
            state_dict = self.state_dict()
            for key in non_lllite_unet_sd:
                if key in state_dict:
                    state_dict[key] = non_lllite_unet_sd[key]
            info = self.load_state_dict(state_dict, False)
            return info

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        # module_name = module_name.replace("_block", "@blocks")
        # module_name = module_name.replace("_layer", "@layer")
        # module_name = module_name.replace("to_", "to@")
        # module_name = module_name.replace("time_embed", "time@embed")
        # module_name = module_name.replace("label_emb", "label@emb")
        # module_name = module_name.replace("skip_connection", "skip@connection")
        # module_name = module_name.replace("proj_in", "proj@in")
        # module_name = module_name.replace("proj_out", "proj@out")
        pattern = re.compile(r"(_block|_layer|to_|time_embed|label_emb|skip_connection|proj_in|proj_out)")

        # convert to lllite with U-Net state dict
        state_dict = non_lllite_unet_sd.copy() if non_lllite_unet_sd is not None else {}
        for key in weights_sd.keys():
            # split with "."
            pos = key.find(".")
            if pos < 0:
                continue

            module_name = key[:pos]
            weight_name = key[pos + 1 :]  # exclude "."
            module_name = module_name.replace(SdxlUNet2DConditionModelControlNetLLLite.LLLITE_PREFIX + "_", "")

            # これはうまくいかない。逆変換を考えなかった設計が悪い / this does not work well. bad design because I didn't think about inverse conversion
            # module_name = module_name.replace("_", ".")

            # ださいけどSDXLのU-Netの "_" を "@" に変換する / ugly but convert "_" of SDXL U-Net to "@"
            matches = pattern.findall(module_name)
            if matches is not None:
                for m in matches:
                    print(module_name, m)
                    module_name = module_name.replace(m, m.replace("_", "@"))
            module_name = module_name.replace("_", ".")
            module_name = module_name.replace("@", "_")

            lllite_key = module_name + ".lllite_" + weight_name

            state_dict[lllite_key] = weights_sd[key]

        info = self.load_state_dict(state_dict, False)
        return info

    def forward(self, x, timesteps=None, context=None, y=None, cond_image=None, **kwargs):
        for m in self.lllite_modules:
            m.set_cond_image(cond_image)
        return super().forward(x, timesteps, context, y, **kwargs)


def replace_unet_linear_and_conv2d():
    print("replace torch.nn.Linear and torch.nn.Conv2d to LLLiteLinear and LLLiteConv2d in U-Net")
    sdxl_original_unet.torch.nn.Linear = LLLiteLinear
    sdxl_original_unet.torch.nn.Conv2d = LLLiteConv2d


if __name__ == "__main__":
    # デバッグ用 / for debug

    # sdxl_original_unet.USE_REENTRANT = False
    replace_unet_linear_and_conv2d()

    # test shape etc
    print("create unet")
    unet = SdxlUNet2DConditionModelControlNetLLLite()

    print("enable ControlNet-LLLite")
    unet.apply_lllite(32, 64, None, False, 1.0)
    unet.to("cuda")  # .to(torch.float16)

    # from safetensors.torch import load_file

    # model_sd = load_file(r"E:\Work\SD\Models\sdxl\sd_xl_base_1.0_0.9vae.safetensors")
    # unet_sd = {}

    # # copy U-Net keys from unet_state_dict to state_dict
    # prefix = "model.diffusion_model."
    # for key in model_sd.keys():
    #     if key.startswith(prefix):
    #         converted_key = key[len(prefix) :]
    #         unet_sd[converted_key] = model_sd[key]

    # info = unet.load_lllite_weights("r:/lllite_from_unet.safetensors", unet_sd)
    # print(info)

    # print(unet)

    # print number of parameters
    params = unet.prepare_params()
    print("number of parameters", sum(p.numel() for p in params))
    # print("type any key to continue")
    # input()

    unet.set_use_memory_efficient_attention(True, False)
    unet.set_gradient_checkpointing(True)
    unet.train()  # for gradient checkpointing

    # # visualize
    # import torchviz
    # print("run visualize")
    # controlnet.set_control(conditioning_image)
    # output = unet(x, t, ctx, y)
    # print("make_dot")
    # image = torchviz.make_dot(output, params=dict(controlnet.named_parameters()))
    # print("render")
    # image.format = "svg" # "png"
    # image.render("NeuralNet") # すごく時間がかかるので注意 / be careful because it takes a long time
    # input()

    import bitsandbytes

    optimizer = bitsandbytes.adam.Adam8bit(params, 1e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("start training")
    steps = 10
    batch_size = 1

    sample_param = [p for p in unet.named_parameters() if ".lllite_up." in p[0]][0]
    for step in range(steps):
        print(f"step {step}")

        conditioning_image = torch.rand(batch_size, 3, 1024, 1024).cuda() * 2.0 - 1.0
        x = torch.randn(batch_size, 4, 128, 128).cuda()
        t = torch.randint(low=0, high=10, size=(batch_size,)).cuda()
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, sdxl_original_unet.ADM_IN_CHANNELS).cuda()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = unet(x, t, ctx, y, conditioning_image)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(sample_param)

    # from safetensors.torch import save_file

    # print("save weights")
    # unet.save_lllite_weights("r:/lllite_from_unet.safetensors", torch.float16, None)
