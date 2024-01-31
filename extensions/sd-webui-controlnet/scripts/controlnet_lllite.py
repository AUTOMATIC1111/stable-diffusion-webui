# https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI/blob/main/node_control_net_lllite.py

import re
import torch

from modules import devices


class LLLiteModule(torch.nn.Module):
    def __init__(
        self,
        name: str,
        is_conv2d: bool,
        in_dim: int,
        depth: int,
        cond_emb_dim: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.name = name
        self.is_conv2d = is_conv2d
        self.is_first = False

        modules = []
        modules.append(torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size*2
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

        self.depth = depth
        self.cond_image = None
        self.cond_emb = None

    def set_cond_image(self, cond_image):
        self.cond_image = cond_image
        self.cond_emb = None

    def forward(self, x, blk_shape):
        if self.cond_emb is None:
            # print(f"cond_emb is None, {self.name}")
            cx = self.conditioning1(self.cond_image.to(x.device, dtype=x.dtype))

            if blk_shape is not None:
                b, c, h, w = blk_shape
                cx = torch.nn.functional.interpolate(cx, (h, w), mode="nearest-exact")

            if not self.is_conv2d:
                # reshape / b,c,h,w -> b,h*w,c
                n, c, h, w = cx.shape
                cx = cx.view(n, c, h * w).permute(0, 2, 1)
            self.cond_emb = cx

        cx = self.cond_emb

        # uncond/condでxはバッチサイズが2倍
        if x.shape[0] != cx.shape[0]:
            if self.is_conv2d:
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1, 1)
            else:
                # print("x.shape[0] != cx.shape[0]", x.shape[0], cx.shape[0])
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        return cx


all_hack = {}


def clear_all_lllite():
    global all_hack
    for k, v in all_hack.items():
        k.forward = v
        k.lllite_list = []
    all_hack = {}
    return


class PlugableControlLLLite(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.cache = {}

        module_weights = {}
        for key, value in state_dict.items():
            fragments = key.split(".")
            module_name = fragments[0]
            weight_name = ".".join(fragments[1:])

            if module_name not in module_weights:
                module_weights[module_name] = {}
            module_weights[module_name][weight_name] = value

        modules = {}
        for module_name, weights in module_weights.items():
            if "conditioning1.4.weight" in weights:
                depth = 3
            elif weights["conditioning1.2.weight"].shape[-1] == 4:
                depth = 2
            else:
                depth = 1

            module = LLLiteModule(
                name=module_name,
                is_conv2d=weights["down.0.weight"].ndim == 4,
                in_dim=weights["down.0.weight"].shape[1],
                depth=depth,
                cond_emb_dim=weights["conditioning1.0.weight"].shape[0] * 2,
                mlp_dim=weights["down.0.weight"].shape[0],
            )
            info = module.load_state_dict(weights)
            modules[module_name] = module
            setattr(self, module_name, module)
            if len(modules) == 1:
                module.is_first = True

        self.modules = modules
        return

    def reset(self):
        self.cache = {}
        return

    @torch.no_grad()
    def hook(self, model, cond, weight, start, end):
        global all_hack

        cond_image = cond * 2.0 - 1.0

        for module in self.modules.values():
            module.set_cond_image(cond_image)

        for k, v in self.modules.items():
            k = k.replace('middle_block', 'middle_blocks_0')
            match = re.match("lllite_unet_(.*)_blocks_(.*)_1_transformer_blocks_(.*)_(.*)_to_(.*)", k, re.M | re.I)
            assert match, 'Failed to load ControlLLLite!'
            root = match.group(1)
            block = match.group(2)
            block_number = match.group(3)
            attn_name = match.group(4)
            proj_name = match.group(5)
            if root == 'input':
                b = model.input_blocks[int(block)][1].transformer_blocks[int(block_number)]
            elif root == 'output':
                b = model.output_blocks[int(block)][1].transformer_blocks[int(block_number)]
            else:
                b = model.middle_block[1].transformer_blocks[int(block_number)]
            b = getattr(b, attn_name, None)
            assert b is not None, 'Failed to load ControlLLLite!'
            b = getattr(b, 'to_' + proj_name, None)
            assert b is not None, 'Failed to load ControlLLLite!'

            if not hasattr(b, 'lllite_list'):
                b.lllite_list = []

            if len(b.lllite_list) == 0:
                all_hack[b] = b.forward
                b.forward = self.get_hacked_forward(original_forward=b.forward, model=model, blk=b)

            b.lllite_list.append((weight, start, end, v))
        return

    def get_hacked_forward(self, original_forward, model, blk):
        @torch.no_grad()
        def forward(x, **kwargs):
            current_sampling_percent = getattr(model, 'current_sampling_percent', 0.5)
            current_h_shape = getattr(model, 'current_h_shape', None)
            is_in_high_res_fix = getattr(model, 'is_in_high_res_fix', False)

            if not is_in_high_res_fix:
                hack = 0
                for weight, start, end, module in blk.lllite_list:
                    module.to(x.device)
                    if current_sampling_percent < start or current_sampling_percent > end:
                        hack = hack + 0
                    else:
                        hack = hack + module(x, current_h_shape) * weight

                x = x + hack

            return original_forward(x, **kwargs)
        return forward
