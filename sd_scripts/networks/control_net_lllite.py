import os
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


class LLLiteModule(torch.nn.Module):
    def __init__(self, depth, cond_emb_dim, name, org_module, mlp_dim, dropout=None, multiplier=1.0):
        super().__init__()

        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"
        self.lllite_name = name
        self.cond_emb_dim = cond_emb_dim
        self.org_module = [org_module]
        self.dropout = dropout
        self.multiplier = multiplier

        if self.is_conv2d:
            in_dim = org_module.in_channels
        else:
            in_dim = org_module.in_features

        # conditioning1はconditioning imageを embedding する。timestepごとに呼ばれない
        # conditioning1 embeds conditioning image. it is not called for each timestep
        modules = []
        modules.append(torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size
        if depth == 1:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
        elif depth == 2:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
        elif depth == 3:
            # kernel size 8は大きすぎるので、4にする / kernel size 8 is too large, so set it to 4
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

        self.conditioning1 = torch.nn.Sequential(*modules)

        # downで入力の次元数を削減する。LoRAにヒントを得ていることにする
        # midでconditioning image embeddingと入力を結合する
        # upで元の次元数に戻す
        # これらはtimestepごとに呼ばれる
        # reduce the number of input dimensions with down. inspired by LoRA
        # combine conditioning image embedding and input with mid
        # restore to the original dimension with up
        # these are called for each timestep

        if self.is_conv2d:
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            # midの前にconditioningをreshapeすること / reshape conditioning before mid
            self.down = torch.nn.Sequential(
                torch.nn.Linear(in_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim, in_dim),
            )

        # Zero-Convにする / set to Zero-Conv
        torch.nn.init.zeros_(self.up[0].weight)  # zero conv

        self.depth = depth  # 1~3
        self.cond_emb = None
        self.batch_cond_only = False  # Trueなら推論時のcondにのみ適用する / if True, apply only to cond at inference
        self.use_zeros_for_batch_uncond = False  # Trueならuncondのconditioningを0にする / if True, set uncond conditioning to 0

        # batch_cond_onlyとuse_zeros_for_batch_uncondはどちらも適用すると生成画像の色味がおかしくなるので実際には使えそうにない
        # Controlの種類によっては使えるかも
        # both batch_cond_only and use_zeros_for_batch_uncond make the color of the generated image strange, so it doesn't seem to be usable in practice
        # it may be available depending on the type of Control

    def set_cond_image(self, cond_image):
        r"""
        中でモデルを呼び出すので必要ならwith torch.no_grad()で囲む
        / call the model inside, so if necessary, surround it with torch.no_grad()
        """
        if cond_image is None:
            self.cond_emb = None
            return

        # timestepごとに呼ばれないので、あらかじめ計算しておく / it is not called for each timestep, so calculate it in advance
        # print(f"C {self.lllite_name}, cond_image.shape={cond_image.shape}")
        cx = self.conditioning1(cond_image)
        if not self.is_conv2d:
            # reshape / b,c,h,w -> b,h*w,c
            n, c, h, w = cx.shape
            cx = cx.view(n, c, h * w).permute(0, 2, 1)
        self.cond_emb = cx

    def set_batch_cond_only(self, cond_only, zeros):
        self.batch_cond_only = cond_only
        self.use_zeros_for_batch_uncond = zeros

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def forward(self, x):
        r"""
        学習用の便利forward。元のモジュールのforwardを呼び出す
        / convenient forward for training. call the forward of the original module
        """
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        cx = self.cond_emb

        if not self.batch_cond_only and x.shape[0] // 2 == cx.shape[0]:  # inference only
            cx = cx.repeat(2, 1, 1, 1) if self.is_conv2d else cx.repeat(2, 1, 1)
            if self.use_zeros_for_batch_uncond:
                cx[0::2] = 0.0  # uncond is zero
        # print(f"C {self.lllite_name}, x.shape={x.shape}, cx.shape={cx.shape}")

        # downで入力の次元数を削減し、conditioning image embeddingと結合する
        # 加算ではなくchannel方向に結合することで、うまいこと混ぜてくれることを期待している
        # down reduces the number of input dimensions and combines it with conditioning image embedding
        # we expect that it will mix well by combining in the channel direction instead of adding

        cx = torch.cat([cx, self.down(x if not self.batch_cond_only else x[1::2])], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)

        if self.dropout is not None and self.training:
            cx = torch.nn.functional.dropout(cx, p=self.dropout)

        cx = self.up(cx) * self.multiplier

        # residual (x) を加算して元のforwardを呼び出す / add residual (x) and call the original forward
        if self.batch_cond_only:
            zx = torch.zeros_like(x)
            zx[1::2] += cx
            cx = zx

        x = self.org_forward(x + cx)  # ここで元のモジュールを呼び出す / call the original module here
        return x


class ControlNetLLLite(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]

    def __init__(
        self,
        unet: sdxl_original_unet.SdxlUNet2DConditionModel,
        cond_emb_dim: int = 16,
        mlp_dim: int = 16,
        dropout: Optional[float] = None,
        varbose: Optional[bool] = False,
        multiplier: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        # self.unets = [unet]

        def create_modules(
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
            module_class: Type[object],
        ) -> List[torch.nn.Module]:
            prefix = "lllite_unet"

            modules = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"

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

                            module = module_class(
                                depth,
                                cond_emb_dim,
                                lllite_name,
                                child_module,
                                mlp_dim,
                                dropout=dropout,
                                multiplier=multiplier,
                            )
                            modules.append(module)
            return modules

        target_modules = ControlNetLLLite.UNET_TARGET_REPLACE_MODULE
        if not TRANSFORMER_ONLY:
            target_modules = target_modules + ControlNetLLLite.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        # create module instances
        self.unet_modules: List[LLLiteModule] = create_modules(unet, target_modules, LLLiteModule)
        print(f"create ControlNet LLLite for U-Net: {len(self.unet_modules)} modules.")

    def forward(self, x):
        return x  # dummy

    def set_cond_image(self, cond_image):
        r"""
        中でモデルを呼び出すので必要ならwith torch.no_grad()で囲む
        / call the model inside, so if necessary, surround it with torch.no_grad()
        """
        for module in self.unet_modules:
            module.set_cond_image(cond_image)

    def set_batch_cond_only(self, cond_only, zeros):
        for module in self.unet_modules:
            module.set_batch_cond_only(cond_only, zeros)

    def set_multiplier(self, multiplier):
        for module in self.unet_modules:
            module.multiplier = multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self):
        print("applying LLLite for U-Net...")
        for module in self.unet_modules:
            module.apply_to()
            self.add_module(module.lllite_name, module)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return False

    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        raise NotImplementedError()

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_optimizer_params(self):
        self.requires_grad_(True)
        return self.parameters()

    def prepare_grad_etc(self):
        self.requires_grad_(True)

    def on_epoch_start(self):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

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


if __name__ == "__main__":
    # デバッグ用 / for debug

    # sdxl_original_unet.USE_REENTRANT = False

    # test shape etc
    print("create unet")
    unet = sdxl_original_unet.SdxlUNet2DConditionModel()
    unet.to("cuda").to(torch.float16)

    print("create ControlNet-LLLite")
    control_net = ControlNetLLLite(unet, 32, 64)
    control_net.apply_to()
    control_net.to("cuda")

    print(control_net)

    # print number of parameters
    print("number of parameters", sum(p.numel() for p in control_net.parameters() if p.requires_grad))

    input()

    unet.set_use_memory_efficient_attention(True, False)
    unet.set_gradient_checkpointing(True)
    unet.train()  # for gradient checkpointing

    control_net.train()

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

    optimizer = bitsandbytes.adam.Adam8bit(control_net.prepare_optimizer_params(), 1e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("start training")
    steps = 10

    sample_param = [p for p in control_net.named_parameters() if "up" in p[0]][0]
    for step in range(steps):
        print(f"step {step}")

        batch_size = 1
        conditioning_image = torch.rand(batch_size, 3, 1024, 1024).cuda() * 2.0 - 1.0
        x = torch.randn(batch_size, 4, 128, 128).cuda()
        t = torch.randint(low=0, high=10, size=(batch_size,)).cuda()
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, sdxl_original_unet.ADM_IN_CHANNELS).cuda()

        with torch.cuda.amp.autocast(enabled=True):
            control_net.set_cond_image(conditioning_image)

            output = unet(x, t, ctx, y)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(sample_param)

    # from safetensors.torch import save_file

    # save_file(control_net.state_dict(), "logs/control_net.safetensors")
