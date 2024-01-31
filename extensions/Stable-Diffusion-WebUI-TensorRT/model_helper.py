#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import json
import tempfile
from typing import List, Tuple

import torch
import numpy as np
import onnx
from onnx import shape_inference, numpy_helper
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

from modules import sd_hijack, sd_unet

from datastructures import ProfileSettings


class UNetModel(torch.nn.Module):
    def __init__(
        self, unet, embedding_dim: int, text_minlen: int = 77, is_xl: bool = False
    ) -> None:
        super().__init__()
        self.unet = unet
        self.is_xl = is_xl

        self.text_minlen = text_minlen
        self.embedding_dim = embedding_dim
        self.num_xl_classes = 2816  # Magic number for num_classes
        self.emb_chn = 1280
        self.in_channels = self.unet.in_channels

        self.dyn_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B", 1: "77N"},
            "timesteps": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
            "y": {0: "2B"},
        }

    def apply_torch_model(self):
        def disable_checkpoint(self):
            if getattr(self, "use_checkpoint", False) == True:
                self.use_checkpoint = False
            if getattr(self, "checkpoint", False) == True:
                self.checkpoint = False

        self.unet.apply(disable_checkpoint)
        self.set_unet("None")

    def set_unet(self, ckpt: str):
        # TODO test if using this with TRT works
        sd_unet.apply_unet(ckpt)
        sd_hijack.model_hijack.apply_optimizations(ckpt)

    def get_input_names(self) -> List[str]:
        names = ["sample", "timesteps", "encoder_hidden_states"]
        if self.is_xl:
            names.append("y")
        return names

    def get_output_names(self) -> List[str]:
        return ["latent"]

    def get_dynamic_axes(self) -> dict:
        io_names = self.get_input_names() + self.get_output_names()
        dyn_axes = {name: self.dyn_axes[name] for name in io_names}
        return dyn_axes

    def get_sample_input(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        text_len: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor]:
        return (
            torch.randn(
                batch_size,
                self.in_channels,
                latent_height,
                latent_width,
                dtype=dtype,
                device=device,
            ),
            torch.randn(batch_size, dtype=dtype, device=device),
            torch.randn(
                batch_size,
                text_len,
                self.embedding_dim,
                dtype=dtype,
                device=device,
            ),
            torch.randn(batch_size, self.num_xl_classes, dtype=dtype, device=device)
            if self.is_xl
            else None,
        )

    def get_input_profile(self, profile: ProfileSettings) -> dict:
        min_batch, opt_batch, max_batch = profile.get_a1111_batch_dim()
        (
            min_latent_height,
            latent_height,
            max_latent_height,
            min_latent_width,
            latent_width,
            max_latent_width,
        ) = profile.get_latent_dim()

        shape_dict = {
            "sample": [
                (min_batch, self.unet.in_channels, min_latent_height, min_latent_width),
                (opt_batch, self.unet.in_channels, latent_height, latent_width),
                (max_batch, self.unet.in_channels, max_latent_height, max_latent_width),
            ],
            "timesteps": [(min_batch,), (opt_batch,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, profile.t_min, self.embedding_dim),
                (opt_batch, profile.t_opt, self.embedding_dim),
                (max_batch, profile.t_max, self.embedding_dim),
            ],
        }
        if self.is_xl:
            shape_dict["y"] = [
                (min_batch, self.num_xl_classes),
                (opt_batch, self.num_xl_classes),
                (max_batch, self.num_xl_classes),
            ]

        return shape_dict

    # Helper utility for weights map
    def export_weights_map(self, onnx_opt_path: str, weights_map_path: dict):
        onnx_opt_dir = onnx_opt_path
        state_dict = self.unet.state_dict()
        onnx_opt_model = onnx.load(onnx_opt_path)

        # Create initializer data hashes
        def init_hash_map(onnx_opt_model):
            initializer_hash_mapping = {}
            for initializer in onnx_opt_model.graph.initializer:
                initializer_data = numpy_helper.to_array(
                    initializer, base_dir=onnx_opt_dir
                ).astype(np.float16)
                initializer_hash = hash(initializer_data.data.tobytes())
                initializer_hash_mapping[initializer.name] = (
                    initializer_hash,
                    initializer_data.shape,
                )
            return initializer_hash_mapping

        initializer_hash_mapping = init_hash_map(onnx_opt_model)

        weights_name_mapping = {}
        weights_shape_mapping = {}
        # set to keep track of initializers already added to the name_mapping dict
        initializers_mapped = set()
        for wt_name, wt in state_dict.items():
            # get weight hash
            wt = wt.cpu().detach().numpy().astype(np.float16)
            wt_hash = hash(wt.data.tobytes())
            wt_t_hash = hash(np.transpose(wt).data.tobytes())

            for initializer_name, (
                initializer_hash,
                initializer_shape,
            ) in initializer_hash_mapping.items():
                # Due to constant folding, some weights are transposed during export
                # To account for the transpose op, we compare the initializer hash to the
                # hash for the weight and its transpose
                if wt_hash == initializer_hash or wt_t_hash == initializer_hash:
                    # The assert below ensures there is a 1:1 mapping between
                    # PyTorch and ONNX weight names. It can be removed in cases where 1:many
                    # mapping is found and name_mapping[wt_name] = list()
                    assert initializer_name not in initializers_mapped
                    weights_name_mapping[wt_name] = initializer_name
                    initializers_mapped.add(initializer_name)
                    is_transpose = False if wt_hash == initializer_hash else True
                    weights_shape_mapping[wt_name] = (
                        initializer_shape,
                        is_transpose,
                    )

            # Sanity check: Were any weights not matched
            if wt_name not in weights_name_mapping:
                print(
                    f"[I] PyTorch weight {wt_name} not matched with any ONNX initializer"
                )
        print(
            f"[I] UNet: {len(weights_name_mapping.keys())} PyTorch weights were matched with ONNX initializers"
        )

        assert weights_name_mapping.keys() == weights_shape_mapping.keys()
        with open(weights_map_path, "w") as fp:
            json.dump([weights_name_mapping, weights_shape_mapping], fp)

    @staticmethod
    def optimize(name, onnx_graph, verbose=False):
        opt = Optimizer(onnx_graph, verbose=verbose)
        opt.info(name + ": original")
        opt.cleanup()
        opt.info(name + ": cleanup")
        opt.fold_constants()
        opt.info(name + ": fold constants")
        opt.infer_shapes()
        opt.info(name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ": finished")
        return onnx_opt_graph


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(
            gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True
        )
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, "model.onnx")
            onnx_inferred_path = os.path.join(temp_dir, "inferred.onnx")
            onnx.save_model(
                onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(
                        int(name.split(".")[1].split("/")[0]), hidden_layers
                    )
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[
                    j
                ] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers - 1
                ):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[
                    j
                ] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers - 1
                ):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph
