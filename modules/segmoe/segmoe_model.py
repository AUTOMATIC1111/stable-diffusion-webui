import gc
from collections import OrderedDict
from typing import Any, Dict, Callable
import os
from copy import deepcopy
from math import ceil
import json
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
)
import tqdm
import yaml


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for _name, child in model._modules.items(): # pylint: disable=protected-access
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


# Inspired from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
class SparseMoeBlock(nn.Module):
    def __init__(self, config, experts):
        super().__init__()
        self.hidden_dim = config["hidden_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.out_dim = config.get("out_dim", self.hidden_dim)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([deepcopy(exp) for exp in experts])

    def forward(self, hidden_states: torch.Tensor, scale=None) -> torch.Tensor: # pylint: disable=unused-argument
        batch_size, sequence_length, f_map_sz = hidden_states.shape
        hidden_states = hidden_states.view(-1, f_map_sz)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        _, selected_experts = torch.topk(
            router_logits.sum(dim=0, keepdim=True), self.top_k, dim=1
        )
        routing_weights = F.softmax(
            router_logits[:, selected_experts[0]], dim=1, dtype=torch.float
        )

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for i, expert_idx in enumerate(selected_experts[0].tolist()):
            expert_layer = self.experts[expert_idx]

            current_hidden_states = routing_weights[:, i].view(
                batch_size * sequence_length, -1
            ) * expert_layer(hidden_states)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states = final_hidden_states + current_hidden_states
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, self.out_dim
        )
        return final_hidden_states


def getActivation(activation, name):
    def hook(model, inp, output): # pylint: disable=unused-argument
        activation[name] = inp

    return hook


class SegMoEPipeline:
    def __init__(self, config_or_path, **kwargs) -> Any:
        """
        Instantiates the SegMoEPipeline. SegMoEPipeline implements the Segmind Mixture of Diffusion Experts, efficiently combining Stable Diffusion and Stable Diffusion Xl models.

        Usage:

        from segmoe import SegMoEPipeline
        pipeline = SegMoEPipeline(config_or_path, **kwargs)

        config_or_path: Path to Config or Directory containing SegMoE checkpoint or HF Card of SegMoE Checkpoint.

        Other Keyword Arguments:
        torch_dtype: Data Type to load the pipeline in. (Default: torch.float16)
        variant: Variant of the Model. (Default: fp16)
        device: Device to load the model on. (Default: cuda)
        Other args supported by diffusers.DiffusionPipeline are also supported.

        For more details visit https://github.com/segmind/segmoe.
        """
        self.torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        self.use_safetensors = kwargs.pop("use_safetensors", True)
        self.variant = kwargs.pop("variant", "fp16")
        self.device = kwargs.pop("device", "cuda")
        if os.path.isfile(config_or_path):
            self.load_from_scratch(config_or_path, **kwargs)
        else:
            if not os.path.isdir(config_or_path):
                cached_folder = DiffusionPipeline.download(config_or_path)
            else:
                cached_folder = config_or_path
            unet = self.create_empty(cached_folder)
            unet.load_state_dict(
                safetensors.torch.load_file(
                    f"{cached_folder}/unet/diffusion_pytorch_model.safetensors"
                )
            )
            self.pipe = DiffusionPipeline.from_pretrained(
                cached_folder,
                unet=unet,
                torch_dtype=self.torch_dtype,
                use_safetensors=self.use_safetensors,
            )
            self.pipe.to(self.device)
            self.pipe.unet.to(
                device=self.device,
                dtype=self.torch_dtype,
                memory_format=torch.channels_last,
            )

    def to(self, *args, **kwargs): # TODO added no-op to avoid error
        self.pipe.to(*args, **kwargs)

    def load_from_scratch(self, config: str, **kwargs) -> None:
        # Load Config
        with open(config, "r", encoding='utf8') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        if self.config.get("num_experts", None):
            self.num_experts = self.config["num_experts"]
        else:
            if self.config.get("experts", None):
                self.num_experts = len(self.config["experts"])
            else:
                if self.config.get("loras", None):
                    self.num_experts = len(self.config["loras"])
                else:
                    self.num_experts = 1
        num_experts_per_tok = self.config.get("num_experts_per_tok", 1)
        self.config["num_experts_per_tok"] = num_experts_per_tok
        moe_layers = self.config.get("moe_layers", "attn")
        self.config["moe_layers"] = moe_layers
        # Load Base Model
        if self.config["base_model"].startswith(
            "https://civitai.com/api/download/models/"
        ):
            os.makedirs("base", exist_ok=True)
            if not os.path.isfile("base/model.safetensors"):
                os.system(
                    "wget -O "
                    + "base/model.safetensors"
                    + self.config["base_model"]
                    + " --content-disposition"
                )
            self.config["base_model"] = "base/model.safetensors"
            self.pipe = DiffusionPipeline.from_single_file(
                self.config["base_model"], torch_dtype=self.torch_dtype
            )
        else:
            try:
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.config["base_model"],
                    torch_dtype=self.torch_dtype,
                    use_safetensors=self.use_safetensors,
                    variant=self.variant,
                    **kwargs,
                )
            except Exception:
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.config["base_model"], torch_dtype=self.torch_dtype, **kwargs
                )
        if self.pipe.__class__ == StableDiffusionPipeline:
            self.up_idx_start = 1
            self.up_idx_end = len(self.pipe.unet.up_blocks)
            self.down_idx_start = 0
            self.down_idx_end = len(self.pipe.unet.down_blocks) - 1
        elif self.pipe.__class__ == StableDiffusionXLPipeline:
            self.up_idx_start = 0
            self.up_idx_end = len(self.pipe.unet.up_blocks) - 1
            self.down_idx_start = 1
            self.down_idx_end = len(self.pipe.unet.down_blocks)
        self.config["up_idx_start"] = self.up_idx_start
        self.config["up_idx_end"] = self.up_idx_end
        self.config["down_idx_start"] = self.down_idx_start
        self.config["down_idx_end"] = self.down_idx_end

        # TODO: Add Support for Scheduler Selection
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)

        # Load Experts
        experts = []
        positive = []
        negative = []
        if self.config.get("experts", None):
            for i, exp in enumerate(self.config["experts"]):
                positive.append(exp["positive_prompt"])
                negative.append(exp["negative_prompt"])
                if exp["source_model"].startswith(
                    "https://civitai.com/api/download/models/"
                ):
                    try:
                        if not os.path.isfile(f"expert_{i}/model.safetensors"):
                            os.makedirs(f"expert_{i}", exist_ok=True)
                            if not os.path.isfile(f"expert_{i}/model.safetensors"):
                                os.system(
                                    f"wget {exp['source_model']} -O "
                                    + f"expert_{i}/model.safetensors"
                                    + " --content-disposition"
                                )
                        exp["source_model"] = f"expert_{i}/model.safetensors"
                        expert = DiffusionPipeline.from_single_file(
                            exp["source_model"],
                        ).to(self.device, self.torch_dtype)
                    except Exception as e:
                        print(f"Expert {i} {exp['source_model']} failed to load")
                        print("Error:", e)
                else:
                    try:
                        expert = DiffusionPipeline.from_pretrained(
                            exp["source_model"],
                            torch_dtype=self.torch_dtype,
                            use_safetensors=self.use_safetensors,
                            variant=self.variant,
                            **kwargs,
                        )

                        # TODO: Add Support for Scheduler Selection
                        expert.scheduler = DDPMScheduler.from_config(
                            expert.scheduler.config
                        )
                    except Exception:
                        expert = DiffusionPipeline.from_pretrained(
                            exp["source_model"], torch_dtype=self.torch_dtype, **kwargs
                        )
                        expert.scheduler = DDPMScheduler.from_config(
                            expert.scheduler.config
                        )
                if exp.get("loras", None):
                    for j, lora in enumerate(exp["loras"]):
                        if lora.get("positive_prompt", None):
                            positive[-1] += " " + lora["positive_prompt"]
                        if lora.get("negative_prompt", None):
                            negative[-1] += " " + lora["negative_prompt"]
                        if lora["source_model"].startswith(
                            "https://civitai.com/api/download/models/"
                        ):
                            try:
                                os.makedirs(f"expert_{i}/lora_{i}", exist_ok=True)
                                if not os.path.isfile(
                                    f"expert_{i}/lora_{i}/pytorch_lora_weights.safetensors"
                                ):
                                    os.system(
                                        f"wget {lora['source_model']} -O "
                                        + f"expert_{i}/lora_{j}/pytorch_lora_weights.safetensors"
                                        + " --content-disposition"
                                    )
                                lora["source_model"] = f"expert_{j}/lora_{j}"
                                expert.load_lora_weights(lora["source_model"])
                                if len(exp["loras"]) == 1:
                                    expert.fuse_lora()
                            except Exception as e:
                                print(
                                    f"Expert{i} LoRA {j} {lora['source_model']} failed to load"
                                )
                                print("Error:", e)
                        else:
                            expert.load_lora_weights(lora["source_model"])
                            if len(exp["loras"]) == 1:
                                expert.fuse_lora()
                experts.append(expert)
        else:
            experts = [deepcopy(self.pipe) for _ in range(self.num_experts)]
        if self.config.get("experts", None):
            if self.config.get("loras", None):
                for i, lora in enumerate(self.config["loras"]):
                    if lora["source_model"].startswith(
                        "https://civitai.com/api/download/models/"
                    ):
                        try:
                            os.makedirs(f"lora_{i}", exist_ok=True)
                            if not os.path.isfile(
                                f"lora_{i}/pytorch_lora_weights.safetensors"
                            ):
                                os.system(
                                    f"wget {lora['source_model']} -O "
                                    + f"lora_{i}/pytorch_lora_weights.safetensors"
                                    + " --content-disposition"
                                )
                            lora["source_model"] = f"lora_{i}"
                            self.pipe.load_lora_weights(lora["source_model"])
                            if len(self.config["loras"]) == 1:
                                self.pipe.fuse_lora()
                        except Exception as e:
                            print(f"LoRA {i} {lora['source_model']} failed to load")
                            print("Error:", e)
                    else:
                        self.pipe.load_lora_weights(lora["source_model"])
                        if len(self.config["loras"]) == 1:
                            self.pipe.fuse_lora()
        else:
            if self.config.get("loras", None):
                j = []
                n_loras = len(self.config["loras"])
                i = 0
                positive = [""] * len(experts)
                negative = [""] * len(experts)
                while n_loras:
                    n = ceil(n_loras / len(experts))
                    j += [i] * n
                    n_loras -= n
                    i += 1
                for i, lora in enumerate(self.config["loras"]):
                    positive[j[i]] += lora["positive_prompt"] + " "
                    negative[j[i]] += lora["negative_prompt"] + " "
                    if lora["source_model"].startswith(
                        "https://civitai.com/api/download/models/"
                    ):
                        try:
                            os.makedirs(f"lora_{i}", exist_ok=True)
                            if not os.path.isfile(
                                f"lora_{i}/pytorch_lora_weights.safetensors"
                            ):
                                os.system(
                                    f"wget {lora['source_model']} -O "
                                    + f"lora_{i}/pytorch_lora_weights.safetensors"
                                    + " --content-disposition"
                                )
                            lora["source_model"] = f"lora_{i}"
                            experts[j[i]].load_lora_weights(lora["source_model"])
                            experts[j[i]].fuse_lora()
                        except Exception:
                            print(f"LoRA {i} {lora['source_model']} failed to load")
                    else:
                        experts[j[i]].load_lora_weights(lora["source_model"])
                        experts[j[i]].fuse_lora()

        # Replace FF and Attention Layers with Sparse MoE Layers
        for i in range(self.down_idx_start, self.down_idx_end):
            for j in range(len(self.pipe.unet.down_blocks[i].attentions)):
                for k in range(
                    len(self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks)
                ):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                self.pipe.unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        # FF Layers
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .ff
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].ff = SparseMoeBlock(config, layers)
                    if not moe_layers == "ff":
                        ## Attns
                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": self.num_experts,
                        }
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_q
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_k
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_v
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_q
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                            "out_dim": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                        }
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_k
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_v
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)

        for i in range(self.up_idx_start, self.up_idx_end):
            for j in range(len(self.pipe.unet.up_blocks[i].attentions)):
                for k in range(
                    len(self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks)
                ):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                self.pipe.unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        # FF Layers
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .ff
                                )
                            )
                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].ff = SparseMoeBlock(config, layers)

                    if not moe_layers == "ff":
                        # Attns
                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_q
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_k
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_v
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_q
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "out_dim": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }

                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_k
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_v
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)

        # Routing Weight Initialization
        if self.config.get("init", "hidden") == "hidden":
            gate_params = self.get_gate_params(experts, positive, negative)
            for i in range(self.down_idx_start, self.down_idx_end):
                for j in range(len(self.pipe.unet.down_blocks[i].attentions)):
                    for k in range(
                        len(
                            self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks
                        )
                    ):
                        # FF Layers
                        if not moe_layers == "attn":
                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[k].ff.gate.weight = nn.Parameter(
                                gate_params[f"d{i}a{j}t{k}"]
                            )

                        # Attns
                        if not moe_layers == "ff":
                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_q.gate.weight = nn.Parameter(
                                gate_params[f"sattnqd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_k.gate.weight = nn.Parameter(
                                gate_params[f"sattnkd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_v.gate.weight = nn.Parameter(
                                gate_params[f"sattnvd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_q.gate.weight = nn.Parameter(
                                gate_params[f"cattnqd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_k.gate.weight = nn.Parameter(
                                gate_params[f"cattnkd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_v.gate.weight = nn.Parameter(
                                gate_params[f"cattnvd{i}a{j}t{k}"]
                            )

            for i in range(self.up_idx_start, self.up_idx_end):
                for j in range(len(self.pipe.unet.up_blocks[i].attentions)):
                    for k in range(
                        len(
                            self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks
                        )
                    ):
                        # FF Layers
                        if not moe_layers == "attn":
                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[k].ff.gate.weight = nn.Parameter(
                                gate_params[f"u{i}a{j}t{k}"]
                            )
                        if not moe_layers == "ff":
                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_q.gate.weight = nn.Parameter(
                                gate_params[f"sattnqu{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_k.gate.weight = nn.Parameter(
                                gate_params[f"sattnku{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_v.gate.weight = nn.Parameter(
                                gate_params[f"sattnvu{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_q.gate.weight = nn.Parameter(
                                gate_params[f"cattnqu{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_k.gate.weight = nn.Parameter(
                                gate_params[f"cattnku{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_v.gate.weight = nn.Parameter(
                                gate_params[f"cattnvu{i}a{j}t{k}"]
                            )
        self.config["num_experts"] = len(experts)
        remove_all_forward_hooks(self.pipe.unet)
        try:
            del experts
            del expert
        except Exception:
            pass
        # Move Model to Device
        self.pipe.to(self.device)
        self.pipe.unet.to(
            device=self.device,
            dtype=self.torch_dtype,
            memory_format=torch.channels_last,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Inference the SegMoEPipeline.

        Calls diffusers.DiffusionPipeline forward with the keyword arguments. See https://github.com/segmind/segmoe#usage for detailed usage.
        """
        return self.pipe(*args, **kwds)

    def create_empty(self, path):
        with open(f"{path}/unet/config.json", encoding='utf8') as f:
            config = json.load(f)
        self.config = config["segmoe_config"]
        unet = UNet2DConditionModel.from_config(config)
        num_experts_per_tok = self.config["num_experts_per_tok"]
        num_experts = self.config["num_experts"]
        moe_layers = self.config["moe_layers"]
        self.up_idx_start = self.config["up_idx_start"]
        self.up_idx_end = self.config["up_idx_end"]
        self.down_idx_start = self.config["down_idx_start"]
        self.down_idx_end = self.config["down_idx_end"]
        for i in range(self.down_idx_start, self.down_idx_end):
            for j in range(len(unet.down_blocks[i].attentions)):
                for k in range(
                    len(unet.down_blocks[i].attentions[j].transformer_blocks)
                ):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        # FF Layers
                        layers = [
                            unet.down_blocks[i].attentions[j].transformer_blocks[k].ff
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].ff = SparseMoeBlock(config, layers)
                    if not moe_layers == "ff":
                        ## Attns
                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }

                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                            "out_dim": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                        }
                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)
        for i in range(self.up_idx_start, self.up_idx_end):
            for j in range(len(unet.up_blocks[i].attentions)):
                for k in range(len(unet.up_blocks[i].attentions[j].transformer_blocks)):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        # FF Layers
                        layers = [
                            unet.up_blocks[i].attentions[j].transformer_blocks[k].ff
                        ] * num_experts
                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].ff = SparseMoeBlock(config, layers)

                    if not moe_layers == "ff":
                        # Attns
                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }

                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "out_dim": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }

                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)
        return unet

    def save_pretrained(self, path):
        """
        Save SegMoEPipeline to Disk.

        Usage:
        pipeline.save_pretrained(path)

        Parameters:
        path: Path to Directory to save the model in.
        """
        for param in self.pipe.unet.parameters():
            param.data = param.data.contiguous()
        self.pipe.unet.config["segmoe_config"] = self.config
        self.pipe.save_pretrained(path)
        safetensors.torch.save_file(
            self.pipe.unet.state_dict(),
            f"{path}/unet/diffusion_pytorch_model.safetensors",
        )

    def cast_hook(self, pipe, dicts):
        for i in range(self.down_idx_start, self.down_idx_end):
            for j in range(len(pipe.unet.down_blocks[i].attentions)):
                for k in range(
                    len(pipe.unet.down_blocks[i].attentions[j].transformer_blocks)
                ):
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].ff.register_forward_hook(getActivation(dicts, f"d{i}a{j}t{k}"))

                    ## Down Self Attns
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_q.register_forward_hook(
                        getActivation(dicts, f"sattnqd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_k.register_forward_hook(
                        getActivation(dicts, f"sattnkd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_v.register_forward_hook(
                        getActivation(dicts, f"sattnvd{i}a{j}t{k}")
                    )

                    ## Down Cross Attns

                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_q.register_forward_hook(
                        getActivation(dicts, f"cattnqd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_k.register_forward_hook(
                        getActivation(dicts, f"cattnkd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_v.register_forward_hook(
                        getActivation(dicts, f"cattnvd{i}a{j}t{k}")
                    )

        for i in range(self.up_idx_start, self.up_idx_end):
            for j in range(len(pipe.unet.up_blocks[i].attentions)):
                for k in range(
                    len(pipe.unet.up_blocks[i].attentions[j].transformer_blocks)
                ):
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].ff.register_forward_hook(getActivation(dicts, f"u{i}a{j}t{k}"))
                    ## Up Self Attns
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_q.register_forward_hook(
                        getActivation(dicts, f"sattnqu{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_k.register_forward_hook(
                        getActivation(dicts, f"sattnku{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_v.register_forward_hook(
                        getActivation(dicts, f"sattnvu{i}a{j}t{k}")
                    )

                    ## Up Cross Attns
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_q.register_forward_hook(
                        getActivation(dicts, f"cattnqu{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_k.register_forward_hook(
                        getActivation(dicts, f"cattnku{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_v.register_forward_hook(
                        getActivation(dicts, f"cattnvu{i}a{j}t{k}")
                    )

    @torch.no_grad
    def get_hidden_states(self, model, positive, negative, average: bool = True):
        intermediate = {}
        self.cast_hook(model, intermediate)
        with torch.no_grad():
            _ = model(positive, negative_prompt=negative, num_inference_steps=25)
        hidden = {}
        for key in intermediate:
            hidden_states = intermediate[key][0][-1]
            if average:
                # use average over sequence
                hidden_states = hidden_states.sum(dim=0) / hidden_states.shape[0]
            else:
                # take last value
                hidden_states = hidden_states[:-1]
            hidden[key] = hidden_states.to(self.device)
        del intermediate
        gc.collect()
        torch.cuda.empty_cache()
        return hidden

    @torch.no_grad
    def get_gate_params(
        self,
        experts,
        positive,
        negative,
    ):
        gate_vects = {}
        for i, expert in enumerate(tqdm.tqdm(experts, desc="Expert Prompts")):
            expert.to(self.device)
            expert.unet.to(
                device=self.device,
                dtype=self.torch_dtype,
                memory_format=torch.channels_last,
            )
            hidden_states = self.get_hidden_states(expert, positive[i], negative[i])
            del expert
            gc.collect()
            torch.cuda.empty_cache()
            for h in hidden_states:
                if i == 0:
                    gate_vects[h] = []
                hidden_states[h] /= (
                    hidden_states[h].norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                )
                gate_vects[h].append(hidden_states[h])
        for h in hidden_states:
            gate_vects[h] = torch.stack(
                gate_vects[h], dim=0
            )  # (num_expert, num_layer, hidden_size)
            gate_vects[h].permute(1, 0)

        return gate_vects
